""" Deep Q Network.

This file is analogous to OpenAI baseline's `build_graph` file.

Resources:
 - https://github.com/econti/minimal_dqn/blob/master/main.py
 - https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent/DQN_agent.py
"""
import typing
from dataclasses import dataclass

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

import attackgraph.rl.pytorch_utils as ptu
from attackgraph.rl.modules.mlp import MLP


@gin.configurable
@dataclass(eq=False, repr=False)
class DQN(nn.Module):
    """ DQN modified with an action mask applied using the `training_attacker` flag. """

    is_attacker: bool

    input_size: int
    hidden_sizes: typing.List
    output_size: int

    eps: float = 1.0  # Initial epsilon value.
    parameter_noise: bool = False
    double_q: bool = True
    grad_norm_clipping: typing.Any = None
    gamma: float = 0.99
    lr: float = 5e-4

    gpu: int = None

    def __post_init__(self):
        nn.Module.__init__(self)
        self.q = MLP(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=self.output_size)
        self.q = ptu.send_to_gpu(self.q, self.gpu)
        self.q_target = MLP(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=self.output_size)
        self.q_target = ptu.send_to_gpu(self.q_target, self.gpu)
        self.optimizer = torch.optim.Adam(
            params=self.q.parameters(),
            lr=self.lr)
        # Set target Q network's weights to be the same as the Q network.
        self.update_target_network()

        # TODO(max): Implement `build_act_with_param_noise`.
        assert not self.parameter_noise, "Parameter noise not implemented."

    def forward(self, *args, **kwargs):
        return self.act(*args, **kwargs)

    def act(self, observation, stochastic, update_eps, mask, training_attacker, select_actions: bool = True, return_numpy: bool = True):
        observation = ptu.send_to_gpu(observation, self.gpu)
        if training_attacker:
            mask = ptu.send_to_gpu(mask, self.gpu)

        batch_dim = observation.shape[0]
        device = observation.device

        q_values = self.q(observation)

        # If we're training the attacker, apply a mask to the actions.
        if training_attacker:
            q_values += mask

        # Return the raw Q values if we shouldn't select an action.
        if not select_actions:
            if return_numpy:
                return q_values.detach().numpy()
            else:
                return q_values

        deterministic_actions = torch.argmax(q_values, dim=1)

        # Select actions from the Q values.
        if stochastic:
            random_actions = Uniform(0, 1)
            random_actions = random_actions.sample([batch_dim, self.output_size])
            random_actions = random_actions.to(device)
            if training_attacker:
                random_actions += mask
            random_actions = torch.argmax(random_actions, dim=1)

            # Epsilon greedy action selection.
            choose_random = Uniform(0, 1)
            choose_random = choose_random.sample([batch_dim]) < self.eps
            choose_random = choose_random.to(device)
            actions = torch.where(choose_random, random_actions, deterministic_actions)
        else:
            actions = deterministic_actions

        # Maybe update epsilon.
        if update_eps >= 0:
            self.eps = update_eps

        actions = actions.cpu().numpy()
        return actions

    def update(self, observations, actions, rewards, next_observations, done_mask, importance_weights, mask, training_attacker, summary_writer, t, **kwargs):
        """ Update the model's parameters based off a batch of experiences.

        :param observations:
        :param actions:
        :param rewards:
        :param next_observations:
        :param done_mask:
        :param importance_weights: A per-experience importance weighting.
        :param mask: An mask of the available actions at `next_observation`.
        :param training_attacker: Is the model we are training an attacker.
        :param summary_writer: TensorboardX SummaryWriter to report loss metrics.
        :param t: Current timestep.
        """
        observations = ptu.send_to_gpu(observations, self.gpu)
        actions = ptu.send_to_gpu(actions, self.gpu).long()
        rewards = ptu.send_to_gpu(rewards, self.gpu)
        next_observations = ptu.send_to_gpu(next_observations, self.gpu)
        done_mask = ptu.send_to_gpu(done_mask, self.gpu)
        importance_weights = ptu.send_to_gpu(importance_weights, self.gpu)
        if training_attacker:
            mask = ptu.send_to_gpu(mask, self.gpu)

        log_prefix = "attacker" if training_attacker else "defender"
        def _log_scalar(key, value):
            summary_writer.add_scalar(f"{log_prefix}/{key}", value, t)

        batch_dim = observations.shape[0]
        batch_range = ptu.send_to_gpu(torch.arange(0, batch_dim), self.gpu).long()

        # Target Q value, the return from the current state.
        # For double q:
        #   \hat{a} \gets \argmax_{a_{t+1}} Q(o_{t+1}, a_{t+1} | \theta)
        #   y \gets r_t + \gamma Q(o_{t+1}, \hat{a}_{t+1} | \theta^{-})
        q_next = self.q_target(next_observations)

        if self.double_q:  # https://arxiv.org/abs/1509.06461 Eqn. 4.
            best_actions = self.q(next_observations)
            if training_attacker:
                best_actions += mask
            best_actions = torch.argmax(best_actions, dim=-1)
            q_next = q_next[batch_range, best_actions]

        else:
            q_next = q_next.max(1)[0]

        q_next = (1.0 - done_mask)*q_next
        q_target = rewards + self.gamma * q_next

        # Actual Q value.
        # \hat{y} \gets Q(o_t, a_t | \theta)
        q_pred = self.q(observations)
        q_pred = q_pred[batch_range, actions]

        # Compute the TD error with Huber loss.
        loss = F.smooth_l1_loss(q_pred, q_target.detach(), reduction="mean")
        _log_scalar("loss", loss.item())

        if self.grad_norm_clipping is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_norm_clipping)

        # Perform update on Q network.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """ Copy Q network to target Q network. """
        self.q_target.load_state_dict(self.q.state_dict())
