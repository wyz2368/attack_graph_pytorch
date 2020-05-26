""" Factory function for creating new policies. """
import gin

import attackgraph.rl.dqn.learner_ops as dqn_ops

from .learner import Learner


@gin.configurable
def learner_factory(policy_type: str):

    if policy_type == "DQN":
        return Learner(get_new_policy=dqn_ops.get_dqn_policy)

    elif policy_type == "DQNEncDec":
        return Learner(get_new_policy=dqn_ops.get_dqn_enc_dec_policy)

    else:
        raise ValueError(f"Unknown policy {policy_type}.")
