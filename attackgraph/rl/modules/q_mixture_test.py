""" Test suite for `q_mixture`. """
import numpy as np
import pytest

from q_mixture import QMixture, QMixtureStateFreq
from replay_buffer import ReplayBuffer


def test_q_mixture():
    """ Basic API test QMixture. """
    mixture = np.array([0.5, 0.5])

    def policy_1(select_actions, *args, **kwargs):
        return np.array([[1.0, 2.0], [10.0, 10.0]])

    def policy_2(select_actions, *args, **kwargs):
        return np.array([[0.0, 1.0], [1.0, 0.0]])

    qmix = QMixture(mixture=mixture, q_funcs=[policy_1, policy_2])
    q_vals = qmix()
    np.testing.assert_array_equal(q_vals, [1, 0])


def test_q_mixture_state_freq():
    """ Basic API test for QMixtureStateFreq. """
    mixture = np.array([0.5, 0.5])

    def policy_1(observation, select_actions, *args, **kwargs):
        return np.array([[1.0, 2.0, 0.0], [10.0, 10.0, 0.0]])

    def policy_2(observation, select_actions, *args, **kwargs):
        return np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    buffer_1 = ReplayBuffer(10)
    buffer_1.add(obs_t=[1.0], action=None, reward=0.0, obs_tp1=None, done=None)
    buffer_1.add(obs_t=[2.0], action=None, reward=0.0, obs_tp1=None, done=None)
    buffer_1.add(obs_t=[3.0], action=None, reward=0.0, obs_tp1=None, done=None)

    buffer_2 = ReplayBuffer(10)
    buffer_2.add(obs_t=[1.0], action=None, reward=0.0, obs_tp1=None, done=None)
    buffer_2.add(obs_t=[1.0], action=None, reward=0.0, obs_tp1=None, done=None)
    buffer_2.add(obs_t=[1.0], action=None, reward=0.0, obs_tp1=None, done=None)

    qmix = QMixtureStateFreq(
        mixture=mixture,
        q_funcs=[policy_1, policy_2],
        replay_buffers=[buffer_1, buffer_2])

    # Basic test of normalized state frequency.
    np.testing.assert_array_equal(qmix.state_weights([1.0]), [.25, .75])
    np.testing.assert_array_equal(qmix.state_weights([2.0]), [1.0, 0.0])

    # Test unknown state returns uniform probability.
    np.testing.assert_array_equal(qmix.state_weights([4.0]), [.5, .5])

    # Test action selection.
    np.testing.assert_array_equal(qmix([[2.0], [0.0]]), [1, 0])

    # Verify the matrix multiplication inside `forward` is sound.
    q_vals = [[[0.7, 0.3], [0.5, 0.5]], [[2, .4], [.4, .6]]]  # [A, B, Q].
    q_vals = np.array(q_vals)
    mixture = [0.2, 0.8]  # [Q].
    mixture = np.array(mixture)
    weights = [[1, 0], [0.5, 0.5]]  # [B, Q].
    weights = np.array(weights)

    final_q_vals = np.sum(np.multiply(q_vals*mixture, weights), axis=2)
    true_q_vals = [[.14, .25], [.4, .28]]
    np.testing.assert_array_almost_equal(final_q_vals, true_q_vals)


if __name__ == "__main__":
    pytest.main(["-s", "--pdb"])
