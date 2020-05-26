""" Launch point for learning new best response policies. """
import os.path as osp

import dill
import gin
import torch

import attackgraph.common.file_ops as fp
from attackgraph import settings
from attackgraph.rl import learner_factory


def train(game, identity, opponent_mix_str, epoch, writer, save_path: str = None, scope: str = None):
    """ Train a best response policy.

    :param game:
    :param identity:
    :param opponent_mix_str:
    :param epoch:
    """
    env = game.env
    env.reset_everything()
    env.set_training_flag(identity)

    if identity:  # Training the attacker.
        if len(opponent_mix_str) != len(game.def_str):
            raise ValueError("The length must match while training.")
        env.defender.set_mix_strategy(opponent_mix_str)
        env.defender.set_str_set(game.def_str)
        if save_path is None:
            save_path = osp.join(settings.get_attacker_strategy_dir(), "att_str_epoch" + str(epoch) + ".pkl")

    else:         # Training the defender.
        if len(opponent_mix_str) != len(game.att_str):
            raise ValueError("The length must match while training.")
        env.attacker.set_mix_strategy(opponent_mix_str)
        env.attacker.set_str_set(game.att_str)
        if save_path is None:
            save_path = osp.join(settings.get_defender_strategy_dir(), "def_str_epoch" + str(epoch) + ".pkl")

    name = "attacker" if identity else "defender"
    scope = name if scope is None else scope
    with gin.config_scope(scope):
        learner = learner_factory()
        policy, best_deviation, replay_buffer, report = learner.learn_multi_nets(env, epoch=epoch, writer=writer, game=game)

    torch.save(policy, save_path, pickle_module=dill)
    fp.save_pkl(replay_buffer, save_path[:-4]+".replay_buffer.pkl")
    return best_deviation, report
