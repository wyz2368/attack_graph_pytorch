""" Sample opponent strategy. """
import os.path as osp

import numpy as np
import torch

import attackgraph.common.file_ops as fp
from attackgraph import settings


def sample_strategy_from_mixed(env, str_set, mix_str, identity, str_dict=None):
    """ Sample a pure strategy from a mixed strategy.

    Note: str in str_set should include .pkl.

    :param env:
    :param str_set:
    :param mix_str:
    :param identity:
    :param str_dict:
    """
    assert env.training_flag != identity
    if not len(str_set) == len(mix_str):
        raise ValueError("Length of mixed strategies does not match number of strategies.")

    mix_str = np.array(mix_str)

    if np.sum(mix_str) != 1.0:
        mix_str = mix_str/np.sum(mix_str)

    picked_str = np.random.choice(str_set, p=mix_str)
    # TODO: modification for fast sampling.
    if str_dict is not None:
        return str_dict[picked_str], picked_str

    if not fp.isInName('.pkl', name=picked_str):
        raise ValueError('The strategy picked is not a pickle file.')

    if identity == 0:  # pick a defender's strategy
        path = settings.get_defender_strategy_dir()
    elif identity == 1:
        path = settings.get_attacker_strategy_dir()
    else:
        raise ValueError("identity is neither 0 or 1!")

    if not fp.isExist(osp.join(path, picked_str)):
        raise ValueError('The strategy picked does not exist!')

    if "epoch1.pkl" in picked_str:
        act = fp.load_pkl(osp.join(path, picked_str))
        return act, picked_str

    act = torch.load(osp.join(path, picked_str))
    return act, picked_str
