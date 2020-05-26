""" Given two strategies, what is the expected pay-off. """
import copy
import logging
import multiprocessing
import os.path as osp
import random

import dill as pickle
import gin
import numpy as np
import torch
from absl import flags

import attackgraph.common.file_ops as fp
from attackgraph import settings
from attackgraph.common.cloudpickle_wrapper import CloudpickleWrapper

logger = logging.getLogger(__name__)
FLAGS = flags.FLAGS


def simulate_expanded_game(game, n_processes: int = 1, save_dir: str = None, summary_writer=None):
    """ Simulate the pay-offs in the newly added rows and columns.

    :param game: Empirical game to expand.
    :param save_dir: Optionally save full simulation values to file in this directory.
    """
    logger.info('Begin simulation and modify payoff matrix.')

    env = game.env
    num_episodes = game.num_episodes

    # TODO: add str first and then calculate payoff
    old_dim, old_dim1 = game.dim_payoff_def()
    new_dim, new_dim1 = game.num_str()
    if old_dim != old_dim1 or new_dim != new_dim1:
        raise ValueError("Payoff dimension does not match.")

    def_str_list = game.def_str
    att_str_list = game.att_str

    position_col_list = []
    position_row_list = []
    for i in range(new_dim-1):
        position_col_list.append((i, new_dim-1))
    for j in range(new_dim):
        position_row_list.append((new_dim-1, j))

    att_col = []
    att_row = []
    def_col = []
    def_row = []
    # TODO: check the path is correct
    for pos in position_col_list:
        idx_def, idx_att = pos
        aReward, dReward = simulate_profile(env, game, att_str_list[idx_att], def_str_list[idx_def], num_episodes, n_processes, save_dir=save_dir, summary_writer=summary_writer)
        att_col.append(aReward)
        def_col.append(dReward)

    for pos in position_row_list:
        idx_def, idx_att = pos
        aReward, dReward = simulate_profile(env, game, att_str_list[idx_att], def_str_list[idx_def], num_episodes, n_processes, save_dir=save_dir, summary_writer=summary_writer)
        att_row.append(aReward)
        def_row.append(dReward)

    game.add_col_att(np.reshape(np.round(np.array(att_col), 2), newshape=(len(att_col), 1)))
    game.add_col_def(np.reshape(np.round(np.array(def_col), 2), newshape=(len(att_col), 1)))
    game.add_row_att(np.round(np.array(att_row), 2)[None])
    game.add_row_def(np.round(np.array(def_row), 2)[None])

    logger.info("Done simulation and modify payoff matrix.")
    return game


@gin.configurable
def simulate_profile(env, game, nn_att, nn_def, n_episodes: int, n_processes: int = 2, save_dir: str = None, summary_writer=None, raw_rewards: bool = False, collect_trajectories: bool = False):
    """ Simulate a payoff from two pure-strategies.

    Resources:
     - https://stackoverflow.com/questions/9038711/python-pool-with-worker-processes
     - https://stackoverflow.com/questions/21609595/python-multiprocessing-with-an-updating-queue-and-an-output-queue

    :param game: Empirical game.
    :type game: EmpiricalGame
    :param nn_att: Name of the attacker's strategy. This is the filename of the neural network.
    :type nn_att: str
    :param nn_def: Name of the defenders's strategy. This is the filename of the neural network.
    :type nn_def: str
    :param n_episodes: Number of episodes to simulate.
    :type n_episodes: int
    :param n_processes: Number of processors to run simulations on.
    :type n_processes: int
    :param raw_rewards: Return the full list of rewards instead of the average.
    :type raw_rewards: bool
    """
    logger.debug(f"Simulating profile: {nn_att} vs. {nn_def}.")
    env.set_training_flag(-1)  # Throw errors if some training functionality is used.

    if n_processes == 1:
        # TODO(max)
        if collect_trajectories:
            attacker_rewards, defender_rewards, trajectories = _simulate_in_series(
                env=env,
                game=game,
                nn_att=nn_att,
                nn_def=nn_def,
                n_episodes=n_episodes,
                save_dir=save_dir,
                summary_writer=summary_writer,
                collect_trajectories=collect_trajectories)
        else:
            attacker_rewards, defender_rewards = _simulate_in_series(
                env=env,
                game=game,
                nn_att=nn_att,
                nn_def=nn_def,
                n_episodes=n_episodes,
                save_dir=save_dir,
                summary_writer=summary_writer,
                collect_trajectories=collect_trajectories)

    else:
        attacker_rewards, defender_rewards = _simulate_in_parallel(
            env=env,
            game=game,
            nn_att=nn_att,
            nn_def=nn_def,
            n_processes=n_processes,
            n_episodes=n_episodes,
            save_dir=save_dir,
            summary_writer=summary_writer)

    # Save the reward lists later for distribution visualization.
    cell_name = "{}_v_{}".format(nn_att[:-4], nn_def[:-4])
    if save_dir is not None:
        # We cut off the ".pkl" extension.
        attacker_save_path = osp.join(save_dir, "{}.{}".format(cell_name, "simulation.attacker_rewards.pkl"))
        defender_save_path = osp.join(save_dir, "{}.{}".format(cell_name, "simulation.defender_rewards.pkl"))

        with open(attacker_save_path, "wb") as out_file:
            pickle.dump(attacker_rewards, out_file)
        with open(defender_save_path, "wb") as out_file:
            pickle.dump(defender_rewards, out_file)

    # Plot to tensorboard.
    if summary_writer is not None:
        summary_writer.add_histogram(f"{cell_name}/attacker", attacker_rewards)
        summary_writer.add_histogram(f"{cell_name}/defender", defender_rewards)

    if raw_rewards:
        if collect_trajectories:
            return attacker_rewards, defender_rewards, trajectories
        else:
            return attacker_rewards, defender_rewards

    attacker_rewards = np.round(np.mean(attacker_rewards), 2)
    defender_rewards = np.round(np.mean(defender_rewards), 2)
    if collect_trajectories:
        return attacker_rewards, defender_rewards, trajectories
    else:
        return attacker_rewards, defender_rewards


class SimulationWorker(multiprocessing.Process):
    """ Process responsible for running environment simulations. """

    def __init__(self, queue, def_queue, att_queue, nn_att, nn_def, attacker_dir, defender_dir):
        super(SimulationWorker, self).__init__()
        self.queue = queue
        self.defender_reward_queue = def_queue
        self.attacker_reward_queue = att_queue
        self.nn_att = nn_att
        self.nn_def = nn_def
        self.attacker_dir = attacker_dir
        self.defender_dir = defender_dir

    def run(self):
        # Process jobs.
        for game in iter(self.queue.get, None):
            game = game()  # Unwrap cloudpickle, this is necessary to ensure that non-serializable data can be sent.
            defender_reward, attacker_reward = _run_simulation(game, self.nn_att, self.nn_def, self.attacker_dir, self.defender_dir)
            self.defender_reward_queue.put(defender_reward)
            self.attacker_reward_queue.put(attacker_reward)


def _simulate_in_series(env, game, nn_att, nn_def, n_episodes: int, save_dir: str = None, summary_writer=None, collect_trajectories: bool=False):
    """ Run simulations in series on a single processor. """
    attacker_rewards = np.zeros([n_episodes])
    defender_rewards = np.zeros([n_episodes])

    trajectories = []

    for episode_i in range(n_episodes):
        if collect_trajectories:
            defender_reward, attacker_reward, traj = _run_simulation(
                game,
                nn_att,
                nn_def,
                settings.get_attacker_strategy_dir(),
                settings.get_defender_strategy_dir(),
                collect_trajectories)
            trajectories += [traj]
        else:
            defender_reward, attacker_reward = _run_simulation(
                game,
                nn_att,
                nn_def,
                settings.get_attacker_strategy_dir(),
                settings.get_defender_strategy_dir(),
                collect_trajectories)
        attacker_rewards[episode_i] = attacker_reward
        defender_rewards[episode_i] = defender_reward

    if collect_trajectories:
        return attacker_rewards, defender_rewards, trajectories
    else:
        return attacker_rewards, defender_rewards


def _simulate_in_parallel(env, game, nn_att, nn_def, n_processes: int, n_episodes: int, save_dir: str = None, summary_writer=None):
    """ Run simulations in parallel processes. """
    worker_processes = []
    simulation_request_queue = multiprocessing.SimpleQueue()
    attacker_reward_queue = multiprocessing.SimpleQueue()
    defender_reward_queue = multiprocessing.SimpleQueue()
    # Set-up all the processes.
    for _ in range(n_processes):
        worker_processes += [SimulationWorker(
            simulation_request_queue,
            defender_reward_queue,
            attacker_reward_queue,
            nn_att,
            nn_def,
            settings.get_attacker_strategy_dir(),
            settings.get_defender_strategy_dir())]
        worker_processes[-1].start()
    # Request all simulations.
    for _ in range(n_episodes):
        simulation_request_queue.put(CloudpickleWrapper(game))
    # Send sentinel values to tell processes to cleanly shutdown (1 per worker).
    for _ in range(n_processes):
        simulation_request_queue.put(None)
    for process in worker_processes:
        process.join()

    # Aggregate results.
    attacker_rewards = np.zeros([n_episodes])
    defender_rewards = np.zeros([n_episodes])
    for episode_i in range(n_episodes):
        attacker_rewards[episode_i] = attacker_reward_queue.get()
        defender_rewards[episode_i] = defender_reward_queue.get()

    return attacker_rewards, defender_rewards


def _run_simulation(game, nn_att_saved, nn_def_saved, attacker_dir, defender_dir, collect_trajectories: bool=False):
    """ Simulate a single episode. """
    env = game.env
    env.reset_everything()
    T = env.T
    G = env.G
    _, targetset = env.get_Targets()
    attacker = env.attacker
    defender = env.defender

    aReward = 0
    dReward = 0

    # Load attacker.
    nn_att = copy.copy(nn_att_saved)
    if isinstance(nn_att, np.ndarray):
        str_set = game.att_str
        nn_att = np.random.choice(str_set, p=nn_att)

    path = osp.join(attacker_dir, nn_att)
    if "epoch1.pkl" in nn_att:
        nn_att_act = fp.load_pkl(path)
    else:
        nn_att_act = torch.load(path)

    # Load defender.
    nn_def = copy.copy(nn_def_saved)
    if isinstance(nn_def, np.ndarray):
        str_set = game.def_str
        nn_def = np.random.choice(str_set, p=nn_def)

    path = osp.join(defender_dir, nn_def)
    if "epoch1.pkl" in nn_def:
        nn_def_act = fp.load_pkl(path)
    else:
        nn_def_act = torch.load(path)

    if collect_trajectories:
        traj = []
        exp = {}

    for t in range(T):
        timeleft = T - t

        if collect_trajectories:
            exp["observations"] = {}
            exp["observations"]["attacker"] = attacker.att_obs_constructor(G, timeleft)
            exp["observations"]["defender"] = defender.def_obs_constructor(G, timeleft)

        attacker.att_greedy_action_builder_single(G, timeleft, nn_att_act)
        defender.def_greedy_action_builder_single(G, timeleft, nn_def_act)

        att_action_set = attacker.attact
        def_action_set = defender.defact

        if collect_trajectories:
            exp["actions"] = {}
            exp["actions"]["attacker"] = att_action_set
            exp["actions"]["attacker"] = att_action_set

        for attack in att_action_set:
            if isinstance(attack, tuple):
                # check OR node
                aReward += G.edges[attack]['cost']
                if random.uniform(0, 1) <= G.edges[attack]['actProb']:
                    G.nodes[attack[-1]]['state'] = 1
            else:
                # check AND node
                aReward += G.nodes[attack]['aCost']
                if random.uniform(0, 1) <= G.nodes[attack]['actProb']:
                    G.nodes[attack]['state'] = 1
        # defender's action
        for node in def_action_set:
            G.nodes[node]['state'] = 0
            dReward += G.nodes[node]['dCost']

        for node in targetset:
            if G.nodes[node]['state'] == 1:
                aReward += G.nodes[node]['aReward']
                dReward += G.nodes[node]['dPenalty']
        # logger.info('aRew:', aReward, 'dRew:', dReward)

        # update players' observations
        # update defender's observation
        defender.update_obs(defender.get_def_hadAlert(G))
        defender.save_defact2prev()
        defender.defact.clear()
        # update attacker's observation
        attacker.update_obs(attacker.get_att_isActive(G))
        attacker.attact.clear()

        if collect_trajectories:
            traj += [exp]
            exp = {}

    if collect_trajectories:
        return dReward, aReward, traj
    else:
        return dReward, aReward
