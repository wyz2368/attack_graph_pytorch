""" Evaluate the quality of a QMixture compared to learned best response. """
import logging
import os
import os.path as osp
import shutil
import sys
import typing
import warnings

import dill
import gin
import numpy as np
import torch
from absl import app, flags
from tensorboardX import SummaryWriter

import attackgraph.common.file_ops as fp
import attackgraph.soccer.policies as player2_policies
from attackgraph import empirical_game, settings
from attackgraph.rl.dqn.dqn import DQN
from attackgraph.rl.modules import LazyLoadNN, QMixture, QMixtureStateFreq
from attackgraph.soccer.agent import Agent
from attackgraph.soccer.envs.gridworld_soccer import GridWorldSoccer
from attackgraph.soccer.simulation import simulate_profile
from attackgraph.soccer.training import Trainer
from attackgraph.soccer.wrappers.multi_to_single_agent_wrapper import MultiToSingleAgentWrapper


# Command line flags.
flags.DEFINE_string(
    "env",
    "run_env_B",
    "Environment's name 1.")
flags.DEFINE_string(
    "run_name",
    None,
    "Experiment's run name.")
flags.DEFINE_multi_string(
    "config_files",
    None,
    "Name of the gin config files to use.")
flags.DEFINE_multi_string(
    "config_overrides",
    [],
    "Overrides for gin config values.")
FLAGS = flags.FLAGS
flags.mark_flag_as_required("run_name")
logger = logging.getLogger("attackgraph")


def main(argv):
    """ Run evaluation script.

    :param argv: Command line arguments.
    """
    # Configure information displayed to terminal.
    np.set_printoptions(precision=2)
    warnings.filterwarnings("ignore")

    # Set-up the result directory.
    run_dir = settings.get_run_dir()
    if osp.exists(run_dir):
        print("Cannot resume previously saved run, overwriting data.")
    else:
        os.mkdir(run_dir)

    # Set-up logging.
    logger = logging.getLogger("attackgraph")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []  # absl has a default handler that we need to remove.
    # logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    # Log to terminal.
    terminal_handler = logging.StreamHandler()
    terminal_handler.setFormatter(formatter)
    # Log to file.
    file_handler = logging.FileHandler(osp.join(run_dir, "out.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    # Debug output.
    debug_handler = logging.FileHandler(osp.join(run_dir, "debug.log"))
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    # Register handlers.
    logger.addHandler(terminal_handler)
    logger.addHandler(file_handler)
    logger.addHandler(debug_handler)

    logger.info(f"Saving results to: {run_dir}")

    # Set-up gin configuration.
    gin_files = [osp.join(settings.SRC_DIR, "configs", f"{x}.gin") for x in FLAGS.config_files]
    gin.parse_config_files_and_bindings(
        config_files=gin_files,
        bindings=FLAGS.config_overrides,
        skip_unknown=False)

    # Save program flags.
    with open(osp.join(run_dir, "flags.txt"), "w") as flag_file:
        # We want only flags relevant to this module to be saved, no extra flags.
        # See: https://github.com/abseil/abseil-py/issues/92
        key_flags = FLAGS.get_key_flags_for_module(argv[0])
        key_flags = "\n".join(flag.serialize() for flag in key_flags)
        flag_file.write(key_flags)
    with open(osp.join(run_dir, "config.txt"), "w") as config_file:
        config_file.write(gin.config_str())

    # Properly restrict pytorch to not consume extra resources.
    #  - https://github.com/pytorch/pytorch/issues/975
    #  - https://github.com/ray-project/ray/issues/3609
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    evaluate_qmix([
        player2_policies.Player2v0(),
        player2_policies.Player2v1(),
        player2_policies.Player2v2(),
        player2_policies.Player2v3(),
        player2_policies.Player2v4()])


@gin.configurable
def evaluate_qmix(opponents: typing.List, mixture: typing.List):
    """ . """
    assert len(opponents) == len(mixture)
    name = "player1"
    env = GridWorldSoccer()

    # -------------------------------------------------------------------------
    # Train best-response to each pure-strategy opponent.
    logger.info("Training best-response against each pure-strategy.")
    best_responses = []
    replay_buffers = []
    best_response_paths = []
    for opponent_i, opponent in enumerate(opponents):
        logger.info(f"  - Training against opponent {opponent_i}")
        br_path = osp.join(settings.get_run_dir(), f"v{opponent_i}.best_response.pkl")
        best_response_paths += [br_path]
        with gin.config_scope("pure"):
            response, replay_buffer = _train(
                br_path,
                opponent,
                SummaryWriter(logdir=osp.join(settings.get_run_dir(), f"br_vs_{opponent_i}")))
        best_responses += [response]
        replay_buffers += [replay_buffer]

    # -------------------------------------------------------------------------
    # Simulate the performance of QMixture.
    logger.info("Simulating the performance of the QMixture.")
    qmix = QMixture(mixture=mixture, q_funcs=best_responses)

    # Save policy, for future evaluation.
    qmix_path = osp.join(settings.get_run_dir(), "qmix.pkl")
    torch.save(qmix, qmix_path, pickle_module=dill)

    qmix_rewards = []
    mixed_reward = 0.0
    reward_std = 0.0
    for opponent_i, opponent in enumerate(opponents):
        rewards, _ = simulate_profile(
            env=env,
            nn_att=qmix,
            nn_def=opponent,
            n_episodes=250,
            save_dir=None,
            summary_writer=None,
            raw_rewards=True)

        logger.info(f"  - Opponent {opponent_i} vs. QMix: {np.mean(rewards)}, {np.std(rewards)}")
        qmix_rewards += [rewards]
        mixed_reward += mixture[opponent_i] * np.mean(rewards)
        reward_std += mixture[opponent_i]**2 * np.std(rewards)**2
    reward_std = np.sqrt(reward_std)
    logger.info(f"Expected reward against mixture opponent: {mixed_reward}, {reward_std}")
    dill.dump(mixed_reward, open(osp.join(settings.get_run_dir(), "qmix.simulated_reward.pkl"), "wb"))

    # -------------------------------------------------------------------------
    # Simulate the performance of QMixture with state frequencies.
    """
    logger.info("Simulating the performance of the QMixture with State-Frequency weighting.")
    qmix_statefreq = QMixtureStateFreq(mixture=mixture, q_funcs=best_responses, replay_buffers=replay_buffers)

    # Save policy, for future evaluation.
    qmix_statefreq_path = osp.join(settings.get_run_dir(), "qmix_statefreq.pkl")
    torch.save(qmix_statefreq, qmix_statefreq_path, pickle_module=dill)

    qmix_statefreq_rewards = []
    mixed_statefreq_reward = 0.0
    for opponent_i, opponent in enumerate(opponents):
        rewards, _ = simulate_profile(
            env=env,
            nn_att=qmix_statefreq,
            nn_def=opponent,
            n_episodes=250,
            save_dir=None,
            summary_writer=SummaryWriter(logdir=osp.join(settings.get_run_dir(), f"simulate_statefreq_vs_{opponent_i}")),
            raw_rewards=True)

        logger.info(f"  - Opponent {opponent_i}: {np.mean(rewards)}, {np.std(rewards)}")
        with open(osp.join(settings.get_run_dir(), f"qmix_statefreq.rewards_v{opponent_i}.pkl"), "wb") as outfile:
            dill.dump(rewards, outfile)
        qmix_statefreq_rewards += [rewards]
        mixed_statefreq_reward += mixture[opponent_i] * np.mean(rewards)
    logger.info(f"Expected reward against mixture opponent: {mixed_statefreq_reward}")
    dill.dump(mixed_reward, open(osp.join(settings.get_run_dir(), "qmix_statefreq.simulated_reward.pkl"), "wb"))
    """
    # -------------------------------------------------------------------------
    # Train best-response to opponent mixture.
    logger.info("Training a best-response against the mixture opponent.")
    mixture_br_path = osp.join(settings.get_run_dir(), "mixture.best_response.pkl")
    opponent_agent = Agent(mixture=mixture, policies=opponents)

    with gin.config_scope("mix"):
        mixture_br, _ = _train(
            mixture_br_path,
            opponent_agent,
            SummaryWriter(logdir=osp.join(settings.get_run_dir(), "br_vs_mixture")))

    # -------------------------------------------------------------------------
    # Evaluate the mixture policy against the individual opponent strategies.
    logger.info("Evaluating the best-response trained against mixture opponents on pure-strategy opponents.")

    mix_br_reward = 0.0
    reward_std = 0.0
    for opponent_i, opponent in enumerate(opponents):
        rewards, _ = simulate_profile(
            env=env,
            nn_att=mixture_br,
            nn_def=opponent,
            n_episodes=250,
            save_dir=None,
            summary_writer=None,
            raw_rewards=True)

        logger.info(f"  - Opponent {opponent_i} vs. MixtureBR: {np.mean(rewards)}, {np.std(rewards)}")
        mix_br_reward += mixture[opponent_i] * np.mean(rewards)
        reward_std += mixture[opponent_i]**2 * np.std(rewards)**2
    reward_std = np.sqrt(reward_std)
    logger.info(f"Expected reward for mixture best-response: {mix_br_reward}, {reward_std}")

    # -------------------------------------------------------------------------
    # Evaluate pure-strategy-best-response policies against all opponents (all pure strategy + mixture).
    logger.info("Evaluating pure-strategy-best-response against all opponent policies.")

    response_rewards = {}
    response_std = {}
    for opponent_i, opponent in enumerate(opponents):
        for response_i, best_response in enumerate(best_responses):
            rewards, _ = simulate_profile(
                env=env,
                nn_att=best_response,
                nn_def=opponent,
                n_episodes=250,
                save_dir=None,
                summary_writer=None,
                raw_rewards=True)

            logger.info(f"  - Opponent {opponent_i} vs. Best-Response {response_i}: {np.mean(rewards)}, {np.std(rewards)}")
            if response_i not in response_rewards:
                response_rewards[response_i] = 0.0
                response_std[response_i] = 0.0
            response_rewards[response_i] += mixture[opponent_i] * np.mean(rewards)
            response_std[response_i] += mixture[opponent_i]**2 * np.std(rewards)**2

    for key, value in response_rewards.items():
        logger.info(f"Expected reward of response {key} against mixture: {value}, {np.sqrt(response_std[key])}")
    logger.info("Finished.")


def _train(policy_save_path, opponent, writer):
    env = GridWorldSoccer()
    env = MultiToSingleAgentWrapper(
        env=env,
        agent_id=1,
        opponents={2: opponent})

    save_path = osp.join(settings.get_run_dir(), osp.basename(policy_save_path))
    save_path = save_path[:-4]  # Remove ".pkl".

    trainer = Trainer(policy_ctor=DQN)
    best_response, _, replay_buffer, _ = trainer.run(
        env=env,
        name=osp.basename(policy_save_path),
        writer=writer)

    # Save data to results folder for QMixture.
    torch.save(best_response, f"{save_path}.pkl", pickle_module=dill)
    fp.save_pkl(replay_buffer, f"{save_path}.replay_buffer.pkl")

    return best_response, replay_buffer


if __name__ == "__main__":
    app.run(main)
