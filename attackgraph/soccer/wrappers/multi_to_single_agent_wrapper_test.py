""" Test suite for `multi_to_single_agent_wrapper`. """
import numpy as np
import pytest

from multi_to_single_agent_wrapper import MultiToSingleAgentWrapper


def test_api():
    """ Basic API test. """
    from attackgraph.envs.gridworld_soccer import GridWorldSoccer
    env = GridWorldSoccer()

    def _player2(state, *args, **kwargs):
        return 2

    env = MultiToSingleAgentWrapper(
        env=env,
        agent_id=1,
        opponents={2: _player2})

    _ = env.reset()
    print(env.render())
    s, r, d, i = env.step(3)
    print(env.render())


if __name__ == "__main__":
    pytest.main(["-s", "--pdb"])
