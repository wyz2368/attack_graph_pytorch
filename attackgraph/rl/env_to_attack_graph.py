""" Environment wrapper to follow atttackgraph API utilized in `Learner`. """
import typing
from dataclasses import dataclass

import gin


class DummyAgent(object):
    """ Placeholder agent to match Attacker/Defender API used in Learner. """

    str_set = []

    def sample_and_set_str(self, str_dict):
        pass


@gin.configurable
@dataclass
class EnvToAttackGraph(object):
    """ Wrapper for a standard gym environment to conform to the `DagGenerator`'s API.

    The purpose of this module is to allow us to test learning rl on
    more well known environments.
    """

    # We always pretend to be a defender so we don't need to worry about
    # the mask processing.
    training_flag = 0

    env: typing.Any

    def __post_init__(self):
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.attacker = DummyAgent()
        self.defender = DummyAgent()

    def step(self, action):
        o, r, d, _ = self.env.step(action)
        return o, r, d

    def act_dim_att(self):
        """ Action space dimension. """
        return self.action_space.n

    def act_dim_def(self):
        """ Action space dimension. """
        return self.action_space.n

    def obs_dim_att(self):
        """ Observation space dimension. """
        return self.observation_space.shape[0]

    def obs_dim_def(self):
        """ Observation space dimension. """
        return self.observation_space.shape[0]

    def reset_everything_with_return(self):
        """ Reset environment. """
        return self.env.reset()
