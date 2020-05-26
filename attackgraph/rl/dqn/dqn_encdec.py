""" Deep Q Network.

This file is analogous to OpenAI baseline's `build_graph` file.

Resources:
 - https://github.com/econti/minimal_dqn/blob/master/main.py
"""
import os.path as osp
import typing
from dataclasses import dataclass

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

import attackgraph.rl.pytorch_utils as ptu
import attackgraph.settings as settings
from attackgraph.rl.modules.mlp import MLP


def _maybe_load_last_epochs_model(is_attacker: bool):
    """ Load the last parameters of the last trained best responder, if any.

    :param is_attacker: Load previous attacker's parameters.
    :type is_attacker: bool
    :return: DQNEncDec parameters, or None if no previous parameters.
    """
    # Get the directory where all of this runs models are saved.
    save_dir = None
    filename = None
    if is_attacker:
        save_dir = settings.get_attacker_strategy_dir()
        filename = "att_str_epoch{}.pkl"
    else:
        save_dir = settings.get_defender_strategy_dir()
        filename = "def_str_epoch{}.pkl"
    filepath_template = osp.join(save_dir, filename)

    # The first epoch where we could have a checkpoint to load
    # is from the 2nd epoch. This is because the 1st epoch's
    # model are random non-parameterized functions. Therefore,
    # if we cannot find at least a checkpoint from the 2nd epoch,
    # there will be nothing to load.
    filepath = filepath_template.format(2)
    if not osp.exists(filepath):
        return None

    # Find the most recent save of a model.
    epoch = 2
    prev_epochs_model = None
    while True:
        filepath = filepath_template.format(epoch)

        # If we cannot find a checkpoint for this epoch, return last checkpoint.
        if not osp.exists(filepath):
            assert prev_epochs_model is not None, "Should've been able to load at least 2nd epoch's model."
            return prev_epochs_model

        # Otherwise, update latest checkpoint.
        prev_epochs_model = torch.load(filepath)
        epoch += 1
    return None


@gin.configurable
@dataclass(eq=False, repr=False)
class DQNEncDec(nn.Module):
    """ DQN modified with an action mask applied using the `training_attacker` flag. """

    is_attacker: bool

    state_dim: int
    action_dim: int
    state_embed_dim: int

    hidden_sizes: typing.List
    state_encoder_load_path: str = None

    eps: float = 1.0
    parameter_noise: bool = False
    double_q: bool = True
    grad_norm_clipping: typing.Any = None
    gamma: float = 0.99
    q_lr: float = 5e-4
    encoder_lr: float = 5e-4

    gpu: int = None

    def __post_init__(self):
        nn.Module.__init__(self)

        self.s_encoder = MLP(
            input_size=self.state_dim,
            hidden_sizes=self.hidden_sizes,
            output_size=self.state_embed_dim)
        self.s_encoder = ptu.send_to_gpu(self.s_encoder, self.gpu)
        self.s_decoder = MLP(
            input_size=self.state_embed_dim,
            hidden_sizes=self.hidden_sizes,
            output_size=self.state_dim)
        self.s_decoder = ptu.send_to_gpu(self.s_decoder, self.gpu)

        def _build_q_fn():
            q = MLP(
                input_size=self.state_embed_dim,
                hidden_sizes=[self.state_embed_dim, self.state_embed_dim],
                output_size=self.action_dim)
            q = ptu.send_to_gpu(q, self.gpu)
            return q

        self.q = _build_q_fn()
        self.q_target = _build_q_fn()
        self.mse_loss = nn.MSELoss()
        self.q_optimizer = torch.optim.Adam(
            list(self.q.parameters()),
            lr=self.q_lr)
        self.encoder_optimizer = torch.optim.Adam(
            list(self.s_encoder.parameters()) + list(self.s_decoder.parameters()),
            lr=self.encoder_lr)
        # Set target Q network's weights to be the same as the Q network.
        self.update_target_network()

        # Optionally load old version of state encoder/decoder
        if self.state_encoder_load_path is None:
            # If we didn' specify a particular state AE to load, then check
            # and see if there is a previous epoch's model to load.
            state_ae = _maybe_load_last_epochs_model(is_attacker=self.is_attacker)
        else:
            state_ae = torch.load(self.state_encoder_load_path)

        if state_ae is not None:
            self.s_encoder.load_state_dict(state_ae.s_encoder.state_dict())
            self.s_decoder.load_state_dict(state_ae.s_decoder.state_dict())

        # TODO(max): Implement `build_act_with_param_noise`.
        assert not self.parameter_noise, "Parameter noise not implemented."

    def forward(self, *args, **kwargs):
        return self.act(*args, **kwargs)

    def act(self, observation, stochastic, update_eps, mask, training_attacker, select_actions: bool = True):
        observation = ptu.send_to_gpu(observation, self.gpu)
        if training_attacker:
            mask = ptu.send_to_gpu(mask, self.gpu)

        batch_dim = observation.shape[0]
        device = observation.device

        q_values = self.q(self.s_encoder(observation))

        # If we're training the attacker, apply a mask to the actions.
        if training_attacker:
            q_values += mask

        # Return the raw Q values if we shouldn't select an action.
        if not select_actions:
            return q_values.detach().numpy()

        deterministic_actions = torch.argmax(q_values, dim=1)

        # Select actions from the Q values.
        if stochastic:
            random_actions = Uniform(0, 1)
            random_actions = random_actions.sample([batch_dim, self.action_dim])
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

        return actions.cpu().numpy()

    def update(self, observations, actions, rewards, next_observations, done_mask, importance_weights, mask, training_attacker, summary_writer, t, **kwargs):
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
        q_next = self.q_target(self.s_encoder(next_observations).detach())

        if self.double_q:  # https://arxiv.org/abs/1509.06461 Eqn. 4.
            best_actions = self.q(self.s_encoder(next_observations).detach())
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
        q_pred = self.q(self.s_encoder(observations).detach())
        q_pred = q_pred[batch_range, actions]

        # Compute the TD error with Huber loss.
        dqn_loss = F.smooth_l1_loss(q_pred, q_target.detach(), reduction="mean")
        _log_scalar("dqn_loss", dqn_loss.item())

        # Compute reconstruction loss.
        # Reconstruction loss.
        o_hat = self.s_decoder(self.s_encoder(observations))
        op_hat = self.s_decoder(self.s_encoder(next_observations))
        reconstruction_loss = self.mse_loss(o_hat, observations) + self.mse_loss(op_hat, next_observations)
        _log_scalar("reconstruction_loss", reconstruction_loss.item())

        if self.grad_norm_clipping is not None:
            parameters = list(self.q.parameters())
            parameters += list(self.s_encoder.parameters())
            parameters += list(self.s_decoder.parameters())
            nn.utils.clip_grad_norm_(parameters, self.grad_norm_clipping)

        # Perform update on Q network.
        self.q_optimizer.zero_grad()
        dqn_loss.backward()
        self.q_optimizer.step()

        # Perform encoder update.
        self.encoder_optimizer.zero_grad()
        reconstruction_loss.backward()
        self.encoder_optimizer.step()

    def update_target_network(self):
        """ Copy Q network to target Q network. """
        self.q_target.load_state_dict(self.q.state_dict())
