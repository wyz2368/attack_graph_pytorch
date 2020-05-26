""" Responsible for determining opponent's strategy from their meta-strategy.

This cache is a light wrapper around the Attacker/Defender's meta-strategy that caches
if there was only one strategy in their meta-strategy, so we can avoid calling the
meta-strategy.
"""
import os.path as osp
from dataclasses import dataclass
from typing import Any

import torch

import attackgraph.common.file_ops as fp
from attackgraph import settings


@dataclass
class OpponentSampler(object):

    env: Any
    opponent_identity: int

    def __post_init__(self):
        assert self.env.training_flag != self.opponent_identity
        assert self.opponent_identity == 0 or self.opponent_identity == 1, f"Invalid opponent identity: {self.opponent_identity}."

    def sample(self):
        # The opponent is the attacker.
        if self.opponent_identity:
            strategy_dict = self.load_all_policies(self.env, self.env.attacker.str_set, opp_identity=1)
            self.env.attacker.sample_and_set_str(str_dict=strategy_dict)

        else:
            strategy_dict = self.load_all_policies(self.env, self.env.defender.str_set, opp_identity=0)
            self.env.defender.sample_and_set_str(str_dict=strategy_dict)

    @staticmethod
    def load_all_policies(env, str_set, opp_identity: int):
        """ Load all of the strategies for an agent.

        :param env:
        :param str_set:
        :param opp_identity: ID of the opponent (0/1 defender/attacker).
        :return: Dictionary from strings to `ActWrapper` policies.
        :rtype: dict
        """
        if opp_identity == 0:  # Pick a defender's strategy.
            path = settings.get_defender_strategy_dir()
        elif opp_identity == 1:
            path = settings.get_attacker_strategy_dir()
        else:
            raise ValueError("identity is neither 0 or 1!")

        str_dict = {}
        count = 1

        for picked_str in str_set:

            # The initial policy is a function, so we do not need to load any parameters.
            if count == 1 and "epoch1" in picked_str:
                str_dict[picked_str] = fp.load_pkl(osp.join(path, picked_str))
                count += 1
                continue

            # Load the policies parameters for epoch > 1.
            str_dict[picked_str] = torch.load(osp.join(path, picked_str))

        return str_dict
