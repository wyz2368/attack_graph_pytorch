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
from tqdm import tqdm

import attackgraph.common.file_ops as fp
from attackgraph import settings
from attackgraph.common.cloudpickle_wrapper import CloudpickleWrapper

logger = logging.getLogger(__name__)
FLAGS = flags.FLAGS


@gin.configurable
def simulate_profile(env, nn_att, nn_def, n_episodes: int, save_dir: str = None, summary_writer=None, raw_rewards: bool = False):
    """ Simulate a payoff from two pure-strategies.

    Resources:
     - https://stackoverflow.com/questions/9038711/python-pool-with-worker-processes
     - https://stackoverflow.com/questions/21609595/python-multiprocessing-with-an-updating-queue-and-an-output-queue

    :param game: Empirical game.
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
    player1_rewards = []
    player2_rewards = []

    for _ in range(n_episodes):
        p1_r, p2_r = _run_simulation(env, nn_att, nn_def)
        player1_rewards += [p1_r]
        player2_rewards += [p2_r]

    if raw_rewards:
        return player1_rewards, player2_rewards

    player1_rewards = np.round(np.mean(player1_rewards), 2)
    player2_rewards = np.round(np.mean(player2_rewards), 2)
    return player1_rewards, player2_rewards


def _run_simulation(env, player1, player2, T_limit: int = 100):
    """ Simulate a single episode. """
    p1_return = 0
    p2_return = 0

    obs = env.reset()
    t = 0

    while True:
        a1 = player1(
            observation=obs[1][None],  # Add batch dimension for DQN's tensor processing.
            stochastic=True,
            update_eps=-1,
            mask=None,
            training_attacker=False)
        a2 = player2(
            observation=obs[2],
            stochastic=True,
            update_eps=-1,
            mask=None,
            training_attacker=False)

        obs, rs, d, _ = env.step({1: a1, 2: a2})

        p1_return += rs[1]
        p2_return += rs[2]

        if d:
            break
        if t > T_limit:
            break

        t += 1

    return p1_return, p2_return
