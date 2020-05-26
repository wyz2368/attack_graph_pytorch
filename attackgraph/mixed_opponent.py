""" EGTA: Mixture of Oracles. """
import copy
import logging
import os.path as osp
import typing

import dill
import gin
import numpy as np
import torch
from tensorboardX import SummaryWriter

import attackgraph.common.file_ops as fp
from attackgraph import empirical_game
from attackgraph import gambit_analysis as gambit_ops
from attackgraph import settings, simulation, training, uniform_str_init, util
from attackgraph.policy_distillation import policy_distillation
from attackgraph.rl.dqn import DQN
from attackgraph.rl.modules import LazyLoadNN, QMixture

logger = logging.getLogger(__name__)


@gin.configurable
def run(
        env_name: str,
        n_epochs: int = 50,
        n_processes: int = 1,
        perform_policy_distillation: bool = False,
        policy_ctor: typing.Any = None):
    """ Run Mixture of Oracles.

    :param env_name: Name of the saved environment.
    :type env_name: str
    :param n_epochs: Number of EGTA epochs to run.
    :type n_epochs: int
    """
    if policy_distillation:
        assert policy_ctor is not None, "Must specify constructor if running distillation."

    logger.info("Initializing EGTA-MO.")

    # Load the saved environment pickle.
    logger.info(f"Loading environment: {env_name}")
    env_path = osp.join(settings.get_env_data_dir(), f"{env_name}.pkl")
    if not osp.exists(env_path):
        raise ValueError(f"Cannot find environment: {env_path}")
    env = fp.load_pkl(env_path)
    logger.info(f"Environment loaded from: {env_path}")

    run_dir = settings.get_run_dir()
    epoch_dir = osp.join(run_dir, f"epoch_1")
    writer = SummaryWriter(logdir=epoch_dir)

    # Initialize the environment's internal configuration.
    env.save_graph_copy()
    env.save_mask_copy()
    env.create_players()
    env.create_action_space()
    logger.info(f"Root nodes: {env.get_Roots()}")
    logger.info(f"Or edges: {env.get_ORedges()}")

    # Initialize the empirical game.
    game = empirical_game.EmpiricalGame(env)
    game.env.defender.set_env_belong_to(game.env)
    game.env.attacker.set_env_belong_to(game.env)
    env.defender.set_env_belong_to(env)
    env.attacker.set_env_belong_to(env)

    # During the first epoch we will add random policies to the strategy sets.
    logger.info(f"EGTA Epoch 1.")
    logger.info("Adding random (uniform) policies to strategy set.")
    att_strategy_name = f"att_str_epoch1.pkl"
    torch.save(
        DQN(input_size=env.obs_dim_att(), output_size=env.act_dim_att(), is_attacker=1),
        osp.join(settings.get_attacker_strategy_dir(), att_strategy_name),
        pickle_module=dill)
    game.add_att_str(att_strategy_name)

    def_strategy_name = f"def_str_epoch1.pkl"
    torch.save(
        DQN(input_size=env.obs_dim_def(), output_size=env.act_dim_def(), is_attacker=0),
        osp.join(settings.get_defender_strategy_dir(), def_strategy_name),
        pickle_module=dill)
    game.add_def_str(def_strategy_name)

    # Simulate the payoff for the random policies.
    logger.info("Beginning simulation of random policies.")
    att_reward, def_reward = simulation.simulate_profile(
        env=game.env,
        game=game,
        nn_att=att_strategy_name,
        nn_def=def_strategy_name,
        n_episodes=game.num_episodes,
        n_processes=n_processes,
        save_dir=epoch_dir,
        summary_writer=writer)
    logger.info("Done simulating random policies.")

    # Initialize the empirical reward matrix.
    game.init_payoffmatrix(att_reward, def_reward)
    nash_eq = {
        0: np.array([1], dtype=np.float32),
        1: np.array([1], dtype=np.float32)}
    game.add_nasheq(1, nash_eq)

    # Save a copy of the game data.
    game_path = osp.join(run_dir, "game.pkl")
    fp.save_pkl(game, game_path)

    logger.info("Running EGTA-MO.")
    for epoch_i in range(2, n_epochs+1):  # We do 1-indexing to be compliant with double-oracle code.
        logger.info(f"EGTA Epoch {epoch_i}.")

        # Create directory to save data about each epoch.
        epoch_dir = osp.join(run_dir, f"epoch_{epoch_i}")
        writer = SummaryWriter(logdir=epoch_dir)

        # Strategy set expansion.
        #  - First we will create a a single opponent policy by Q-Mixing their
        #    strategy set following their equilibrium mixed strategy.
        #  - Then we will learn best-responses against the merged opponent.
        nash_eq = game.nasheq[epoch_i - 1]

        # Merge the opponent's strategy set into a single policy.
        att_merged = QMixture(mixture=nash_eq[1], q_funcs=_load_strategy_set(epoch_i, 1))
        def_merged = QMixture(mixture=nash_eq[0], q_funcs=_load_strategy_set(epoch_i, 0))

        # Compress the QMix policies into a smaller parameter space.
        if perform_policy_distillation and epoch_i > 2:
            att_merged = policy_distillation(
                teacher=att_merged,
                student_ctor=gin.REQUIRED,
                env=game.env,
                epoch=epoch_i,
                training_attacker=1)
            def_merged = policy_distillation(
                teacher=def_merged,
                student_ctor=gin.REQUIRED,
                env=game.env,
                epoch=epoch_i,
                training_attacker=0)

        # Now we need to train best-responses against each of the merged opponents.
        # Temporary hack to allow us to train against a single opponent that will not be
        # permanently part of the opponent's strategy set.
        modded_game = copy.deepcopy(game)

        att_merge_name = f"att_qmix_epoch{epoch_i}.pkl"
        torch.save(att_merged, osp.join(settings.get_attacker_strategy_dir(), att_merge_name), pickle_module=dill)
        modded_game.add_att_str(att_merge_name)

        def_merge_name = f"def_qmix_epoch{epoch_i}.pkl"
        torch.save(def_merged, osp.join(settings.get_defender_strategy_dir(), def_merge_name), pickle_module=dill)
        modded_game.add_def_str(def_merge_name)

        opponent_mixture = np.zeros_like(modded_game.def_str, dtype=np.float32)
        opponent_mixture[-1] = 1.0

        logger.info("Training attacker's best-response to the defender's QMix.")
        a_BD, _ = training.train(modded_game, 1, opponent_mixture, epoch_i, writer)
        logger.info("Done training the attacker's new best-response.")

        logger.info("Training defender's best-response to the attacker's QMix.")
        d_BD, _ = training.train(modded_game, 0, opponent_mixture, epoch_i, writer)
        logger.info("Done training the defender's new best-response.")

        game.att_BD_list.append(a_BD)
        game.def_BD_list.append(d_BD)

        game.add_att_str(f"att_str_epoch{epoch_i}.pkl")
        game.add_def_str(f"def_str_epoch{epoch_i}.pkl")

        # Game simulation.
        game = simulation.simulate_expanded_game(
            game=game,
            n_processes=n_processes,
            save_dir=epoch_dir,
            summary_writer=writer)

        # Nash calculations.
        nash_att, nash_def = gambit_ops.do_gambit_analysis(
            game.payoffmatrix_def,
            game.payoffmatrix_att)
        gambit_ops.add_new_NE(game, nash_att, nash_def, epoch_i)
        game.env.attacker.nn_att = None
        game.env.attacker.nn_def = None
        fp.save_pkl(game, game_path)

    logger.info("Job complete.")


def _load_strategy_set(epoch: int, is_attacker: bool) -> typing.List:
    """ Load all of an agent's pure-strategy-opponent best-responses.

    :param epoch: Current epoch.
    :type epoch: int
    :param is_attacker: If the agent is the attacker.
    :type is_attacker: bool
    :return: List of best responder functions.
    :rtype: List
    """
    name = "att_str_epoch{}.pkl" if is_attacker else "def_str_epoch{}.pkl"
    save_dir = settings.get_attacker_strategy_dir() if is_attacker else settings.get_defender_strategy_dir()
    policies = [None] * (epoch - 1)

    for policy_i in range(1, epoch):
        policies[policy_i - 1] = LazyLoadNN(save_path=osp.join(save_dir, name.format(policy_i)))

    return policies
