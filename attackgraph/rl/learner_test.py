""" Test suite for `learner`. """
import sys

import gym
import pytest
from absl import flags

from attackgraph import settings
from attackgraph.rl.env_to_attack_graph import EnvToAttackGraph
from attackgraph.rl.learner import Learner


def test_dqn_cartpole():
    """ Test DQN.

    References:
     - https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn_cartpole.gin
    """
    from attackgraph.rl.dqn import DQN
    import wandb

    flags.DEFINE_string("run_name", "test_dqn_cartpole", "")
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    wandb.init(
        project="test_dqn_cartpole",
        dir=settings.get_run_dir(),
        resume=False)

    def _policy_factory(*args, **kwargs):
        """ Generate new policy. """
        return DQN(
            is_attacker=True,
            state_dim=4,
            hidden_sizes=[8, 4],
            action_dim=2,
            lr=0.0001)

    env = gym.make("CartPole-v0")
    env.seed(0)
    env = EnvToAttackGraph(env)

    trainer = Learner(
        seed=0,
        # Policy.
        get_new_policy=_policy_factory,
        exploration_fraction=0.4,
        exploration_final_eps=0.01,
        # Time.
        total_timesteps=400000,
        learning_starts=500,
        train_freq=4,
        target_network_update_freq=10000,
        gamma=0.9,
        # Replay buffer.
        batch_size=512,
        buffer_size=1000000)
    trainer.learn_multi_nets(env, epoch=0)


if __name__ == "__main__":
    pytest.main(["-s", "--pdb"])
