""" Mixture of Q functions from >=1 Q-functions.

Note: is expected that all of the q-functions will have the same callable interface. This includes
a parameter called `select_actions`, that determines whether the function returns the raw
Q values (false), or an action (true).

The purpose of this class is to provide a mechanism for providing a weighted combination (mixture)
of several Q-functions. We make no assumptions about the underlying Q-functions, so that they may
be either PyTorch modules, Python functions, etc..
"""
import typing
from dataclasses import dataclass

import gin
import numpy as np
from scipy.special import softmax
import torch
import torch.nn as nn


@gin.configurable
@dataclass
class QMixture(nn.Module):
    f"""{__doc__}"""

    mixture: np.ndarray
    q_funcs: typing.List

    def __post_init__(self):
        nn.Module.__init__(self)
        self.mixture = np.array(self.mixture)
        assert len(self.mixture.shape) == 1, "Expected a 1D list mixture."
        assert self.mixture.shape[0] == len(self.q_funcs), "Each Q-function must have a mixing coefficient."

    def forward(self, select_actions: bool=True, *args, **kwargs):
        """ Call all of the q_funcs and mix their output.

        :return:
        :rtype:
        """
        if "select_actions" in kwargs:
            del kwargs["select_actions"]
        q_vals = [q(select_actions=False, *args, **kwargs) for q in self.q_funcs]
        q_vals = [w*q for w, q in zip(self.mixture, q_vals)]
        q_vals = np.sum(q_vals, axis=0)

        if select_actions:
            return np.argmax(q_vals, axis=1)
        else:
            return q_vals


@gin.configurable
@dataclass
class QMixtureStateFreq(nn.Module):
    """ Q-Mixing re-weighted by the state frequency.

    In this module we additionally reweight the pure-strategy learnd best-response by the
    frequency of each observation in the respective pure-strategies' replay buffers
    from the end of their training.
    """

    mixture: np.ndarray
    q_funcs: typing.List
    replay_buffers: typing.List

    def __post_init__(self):
        nn.Module.__init__(self)
        self.mixture = np.array(self.mixture)
        assert len(self.mixture.shape) == 1, "Expected a 1D list mixture."
        assert self.mixture.shape[0] == len(self.q_funcs), "Each Q-function must have a mixing coefficient."
        assert self.mixture.shape[0] == len(self.replay_buffers), "Each Q-function needs a state-freq. "

    def forward(self, observation, *args, **kwargs):
        """ Call all of the q_funcs and mix their output.

        :return:
        :rtype:
        """
        if "select_actions" in kwargs:
            assert kwargs["selection_actions"] is False
            del kwargs["select_actions"]
        q_vals = [q(observation, select_actions=False, *args, **kwargs) for q in self.q_funcs]
        q_vals = np.array(q_vals).T
        weights = [self.state_weights(o) for o in observation]
        # - Weights: [B, Q].
        # - Mixture: [Q].
        # - Q-Vals: [A, B, Q].
        q_vals = np.multiply(q_vals*self.mixture, weights)  # [A, B, Q].
        q_vals = np.sum(q_vals, axis=2)                     # [A, B].
        return np.argmax(q_vals, axis=0)                    # [B].

    def state_weights(self, state):
        """ Get the per-opponent-strategy state-frequency weighting of each state.

        :param state:
        :return:
        """
        state_freqs = np.zeros([len(self.replay_buffers)])

        for buffer_i, buffer in enumerate(self.replay_buffers):
            for experience in buffer._storage:
                exp_state, _, _, _, _ = experience
                if np.all(state == exp_state):
                    state_freqs[buffer_i] += 1

        if np.sum(state_freqs) > 0:
            return state_freqs / np.sum(state_freqs)
        else:
            return np.ones([len(self.replay_buffers)]) / len(self.replay_buffers)


@gin.configurable
class QMixtureWithOpponentClassifier(nn.Module):
    """ Q-Mixing with an opponent classifier.

    The opponent classifier outputs a distribution over the possible opponents, and
    this is used to re-weight the Q-values.
    """

    def __init__(self, mixture, q_funcs, classifier):
        nn.Module.__init__(self)
        self.mixture = mixture
        self.q_funcs = q_funcs
        self.classifier = classifier
        self.mixture = np.array(self.mixture)
        assert len(self.mixture.shape) == 1, "Expected a 1D list mixture."
        assert self.mixture.shape[0] == len(self.q_funcs), "Each Q-function must have a mixing coefficient."

    def forward(self, observation, *args, **kwargs):
        """ Call all of the q_funcs and mix their output.

        :return:
        :rtype:
        """
        if "select_actions" in kwargs:
            assert kwargs["selection_actions"] is False
            del kwargs["select_actions"]
        q_vals = [q(observation, select_actions=False, *args, **kwargs) for q in self.q_funcs]
        q_vals = np.array(q_vals).T
        weights = [self.state_weights(o) for o in observation]
        # - Weights: [B, Q].
        # - Mixture: [Q].
        # - Q-Vals: [A, B, Q].
        q_vals = np.multiply(q_vals*self.mixture, weights)  # [A, B, Q].
        q_vals = np.sum(q_vals, axis=2)                     # [A, B].
        return np.argmax(q_vals, axis=0)                    # [B].

    def state_weights(self, state):
        """ Get the per-opponent-strategy state-frequency weighting of each state.

        :param state:
        :return:
        """
        weights = self.classifier(torch.tensor(state)).detach().numpy()
        weights = softmax(weights)
        return weights


@gin.configurable
@dataclass
class QMixtureSubStateFreq(nn.Module):
    """ Q-Mixing re-weighted by the state frequency, where we only consider a subset of the state.

    Note: This module requires that the sub-state is a contiguous sub-space (i.e., can be sliced).
    """

    mixture: np.ndarray
    q_funcs: typing.List
    replay_buffers: typing.List
    subspace_start: int = 0
    subspace_end: int = -1

    def __post_init__(self):
        nn.Module.__init__(self)
        self.mixture = np.array(self.mixture)
        assert len(self.mixture.shape) == 1, "Expected a 1D list mixture."
        assert self.mixture.shape[0] == len(self.q_funcs), "Each Q-function must have a mixing coefficient."
        assert self.mixture.shape[0] == len(self.replay_buffers), "Each Q-function needs a state-freq. "

    def forward(self, observation, *args, **kwargs):
        """ Call all of the q_funcs and mix their output.

        :return:
        :rtype:
        """
        if "select_actions" in kwargs:
            assert kwargs["selection_actions"] is False
            del kwargs["select_actions"]
        q_vals = [q(observation, select_actions=False, *args, **kwargs) for q in self.q_funcs]
        q_vals = np.array(q_vals).T
        weights = [self.state_weights(o) for o in observation]
        # - Weights: [B, Q].
        # - Mixture: [Q].
        # - Q-Vals: [A, B, Q].
        q_vals = np.multiply(q_vals*self.mixture, weights)  # [A, B, Q].
        q_vals = np.sum(q_vals, axis=2)                     # [A, B].
        return np.argmax(q_vals, axis=0)                    # [B].

    def state_weights(self, state):
        """ Get the per-opponent-strategy state-frequency weighting of each state.

        :param state:
        :return:
        """
        state_freqs = np.zeros([len(self.replay_buffers)])

        for buffer_i, buffer in enumerate(self.replay_buffers):
            for experience in buffer._storage:
                exp_state, _, _, _, _ = experience

                substate = state[self.subspace_start:self.subspace_end]
                to_check = exp_state[self.subspace_start:self.subspace_end]

                if np.all(substate == to_check):
                    state_freqs[buffer_i] += 1

        if np.sum(state_freqs) > 0:
            return state_freqs / np.sum(state_freqs)
        else:
            return np.ones([len(self.replay_buffers)]) / len(self.replay_buffers)


@gin.configurable
@dataclass
class QMixtureHard(QMixtureStateFreq):

    def state_weights(self, state):
        """ Get the per-opponent-strategy state-frequency weighting of each state.

        :param state:
        :return:
        """
        state_freqs = np.zeros([len(self.replay_buffers)])

        for buffer_i, buffer in enumerate(self.replay_buffers):
            for experience in buffer._storage:
                exp_state, _, _, _, _ = experience
                if np.all(state == exp_state):
                    state_freqs[buffer_i] += 1

        if np.sum(state_freqs) > 0:
            # Get all occurrences of the maximum value, and assign them equal weight.
            # https://stackoverflow.com/questions/17568612/how-to-make-numpy-argmax-return-all-occurrences-of-the-maximum
            max_indices = np.argwhere(state_freqs == np.amax(state_freqs))
            max_indices = max_indices.flatten().tolist()

            weights = np.zeros_like(state_freqs)

            # If there is only one maximum value, assign it the full probability.
            if len(max_indices) == 1:
                weights[np.argmax(state_freqs)] = 1

            # Otherwise, we need to assign each index some proportional probability.
            else:
                for index in max_indices:
                    weights[index] = 1.0 / float(len(max_indices))

            return weights
        else:
            return np.ones([len(self.replay_buffers)]) / len(self.replay_buffers)
