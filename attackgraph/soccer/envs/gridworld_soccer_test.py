""" Test suite for `q_mixture`. """
import numpy as np
import pytest

import gridworld_soccer as soccer


def test_api():
    """ Basic API test. """
    env = soccer.GridWorldSoccer()
    print()
    print(env.render())
    print("Player 1 stepping North, and Player 2 stepping South.")
    s, r, d, i = env.step({1: 0, 2: 1})
    np.testing.assert_array_equal(s, [0, 1, 1, 2, 1])
    print(s, r, d, i)
    print(env.render())
    print("Player 1 stepping East, and Player 2 standing still.")
    s, r, d, i = env.step({1: 3, 2: 4})
    np.testing.assert_array_equal(s, [0, 2, 1, 2, 1])
    print(s, r, d, i)
    print(env.render())
    print("Player 1 stepping East, and Player 2 standing still.")
    s, r, d, i = env.step({1: 3, 2: 4})
    np.testing.assert_array_equal(s, [0, 3, 1, 2, 1])
    print(s, r, d, i)
    print(env.render())


if __name__ == "__main__":
    pytest.main(["-s", "--pdb"])
