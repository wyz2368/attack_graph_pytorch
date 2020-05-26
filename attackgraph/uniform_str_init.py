""" Initial policies. """
import numpy as np


def act_att(observation, mask, training_attacker, stochastic=True, update_eps=-1):
    if training_attacker != 1:
        raise ValueError("training flag for uniform att str is not 1")

    legal_action = np.where(mask[0] == 0)[0]
    return [np.random.choice(legal_action)]


def act_def(observation, mask, training_attacker, stochastic=True, update_eps=-1):
    if training_attacker != 0:
        raise ValueError("training flag for uniform def str is not 0")

    legal_action = np.where(mask[0] == 0)[0]
    return [np.random.choice(legal_action)]
