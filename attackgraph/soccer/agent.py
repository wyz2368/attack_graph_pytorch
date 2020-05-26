""" Agent for playing a game.

Meant to deal with strategy set ownership and policy selection.
"""
import typing
from dataclasses import dataclass

import numpy as np
import torch.nn as nn


@dataclass
class Agent(nn.Module):
    f"""{__doc__}"""

    mixture: np.ndarray
    policies: typing.List

    def __post_init__(self):
        nn.Module.__init__(self)
        self.mixture = np.array(self.mixture)
        assert self.mixture.shape[0] == len(self.policies), "Each strategy must have a mixing coefficient."
        self.current_policy_index = None

    def forward(self, *args, **kwargs):
        assert self.current_policy_index is not None, "Must call `begin_episode()` prior to action selection."
        return self.policies[self.current_policy_index](*args, **kwargs)

    def begin_episode(self, *args, **kwargs):
        self.current_policy_index = np.random.choice(
            np.arange(len(self.policies)),
            p=self.mixture)
