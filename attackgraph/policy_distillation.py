""" Policy distillation.


"""
import copy
import logging
import os.path as osp
import tempfile
import typing

import dill
import gin
import numpy as np
import torch
import torch.nn.functional as F
from absl import flags
from tensorboardX import SummaryWriter
from tqdm import tqdm

import attackgraph.common.file_ops as fp
from attackgraph import empirical_game, settings
from attackgraph.rl.modules.q_mixture import QMixture
from attackgraph.rl.modules.replay_buffer import ReplayBuffer
from attackgraph.simulation import simulate_profile
from attackgraph.util import mask_generator_att

logger = logging.getLogger(__name__)
FLAGS = flags.FLAGS


@gin.configurable
def policy_distillation(teacher: QMixture, student_ctor: typing.Any, env: typing.Any, epoch: int, training_attacker: bool):
    """ Entry point for running policy distilation during the Mixed Oracle algorthm.

    :param teacher:
    :param student_ctor:
    :param env:
    :param epoch:
    :training_attacker:
    :return:
    """
    # Build a student, via looking up the correct environment dimensions.
    state_dim = env.act_dim_att() if training_attacker else env.act_dim_def()
    action_dim = env.obs_dim_att() if training_attacker else env.obs_dim_def()
    student = student_ctor(state_dim=state_dim, action_dim=action_dim)

    # Load all the replay buffers and merge/split them.
    buffers = []
    buffer_save_name = "att_br_epoch{}.replay_buffer.pkl" if training_attacker else "def_br_epoch{}.replay_buffer.pkl"
    buffer_save_dir = settings.get_attacker_best_response_dir() if training_attacker else settings.get_defender_best_response_dir()
    for epoch_i in range(2, epoch):
        if teacher.mixture[epoch_i] < 0.000001:
            continue
        buffers += [fp.load_pkl(osp.join(buffer_save_dir, buffer_save_name.format(epoch_i)))]
    replay_buffer = merge_replay_buffers(buffers)

    # Run policy distillation.
    student = distill_policy(
        teacher=teacher,
        student=student,
        env=env,
        dataset=replay_buffer,
        training_attacker=training_attacker)
    return student


@gin.configurable
def distill_policy(teacher, student, env, dataset, training_attacker: bool, n_epochs: int = 1, batch_size: int = 2, learning_rate: float = 0.003, tau: float=0.01, writer=None):
    """ Use a more complex model (teacher) to train a simpler model (student) on the dataset.

    :param teacher:
    :param student:
    :param dataset:
    """
    n_batches_per_epoch = len(dataset) // batch_size
    optimizer = torch.optim.Adam(
        params=student.parameters(),
        lr=learning_rate)

    student.train()
    teacher.eval()

    def _softmax_kl_div(prediction, target):
        """ https://arxiv.org/pdf/1511.06295.pdf """
        target = target / tau
        target = F.softmax(target, dim=1)
        prediction = F.log_softmax(prediction, dim=1)
        loss = F.kl_div(prediction, target)
        return loss

    criterion = _softmax_kl_div

    if writer is None:
        writer = SummaryWriter(logdir=settings.get_run_dir())

    t = 0
    for epoch_i in range(n_epochs):
        # Training.
        logger.info(f"Epoch: {epoch_i}")
        for batch_i in tqdm(range(n_batches_per_epoch)):
            student.train()

            o, a, r, op, d = dataset.sample(batch_size)

            # Generate action masks.
            mask = mask_generator_att(env, op) if training_attacker else None

            # Compute the Q-values for loss calculation.
            target = teacher(
                observation=o,
                stochastic=False,
                update_eps=-1,
                mask=mask,
                training_attacker=training_attacker,
                select_actions=False)
            target = torch.Tensor(target)
            prediction = student(
                observation=o,
                stochastic=False,
                update_eps=-1,
                mask=mask,
                training_attacker=training_attacker,
                select_actions=False,
                return_numpy=False)
            loss = criterion(prediction, target.detach())
            writer.add_scalar("loss", loss.item(), t)

            # Calculate the accuracy of the actions selected.
            with torch.no_grad():
                target_actions = teacher(
                    observation=o,
                    stochastic=False,
                    update_eps=-1,
                    mask=mask,
                    training_attacker=training_attacker,
                    select_actions=True)
                predicted_actions = student(
                    observation=o,
                    stochastic=False,
                    update_eps=-1,
                    mask=mask,
                    training_attacker=training_attacker,
                    select_actions=True,
                    return_numpy=True)
                accuracy = np.mean(target_actions == predicted_actions)
                writer.add_scalar("accuracy", accuracy, t)

            # Calculate the rank of the target action.
            with torch.no_grad():
                target_actions = teacher(
                    observation=o,
                    stochastic=False,
                    update_eps=-1,
                    mask=mask,
                    training_attacker=training_attacker,
                    select_actions=True)
                q_vals = student(
                    observation=o,
                    stochastic=False,
                    update_eps=-1,
                    mask=mask,
                    training_attacker=training_attacker,
                    select_actions=False,
                    return_numpy=True)
                rank_mean, rank_std = calculate_action_rank(target_actions, q_vals)
                writer.add_scalar("action_rank/mean", rank_mean, t)
                writer.add_scalar("action_rank/std", rank_std, t)

            # Calcuate the expected reward
            if t % 1 == 0:
                student.eval()
                with torch.no_grad():
                    reward_mean, reward_std = simulate_rewards(policy=student)
                    writer.add_scalar("reward/mean", reward_mean, t)
                    writer.add_scalar("reward/std", reward_std, t)

            # Update the student from the teacher's training signal.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t += 1

    return student


def calculate_action_rank(targets, predictions):
    """ Calculate the rank of the target indices in the predictions.

    :param targets: [B]\in[0, k).
    :param predictions: [B, k].
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    # Get the predicted values of each of the target actions.
    pred_vals = predictions[np.arange(targets.shape[0]), targets]
    # Copy the values to allow us to perform logical operations tensor-wise.
    pred_vals = np.repeat(np.expand_dims(pred_vals, axis=1), predictions.shape[1], axis=1)
    # Count all of the actions with a higher value.
    better_actions = predictions > pred_vals
    n_better_actions = np.sum(better_actions, axis=1)
    return np.mean(n_better_actions), np.std(n_better_actions)


@gin.configurable
def simulate_rewards(policy, opponent_paths, mixture, is_attacker):
    expected_reward = 0.0
    expected_reward_std = 0.0

    # Save the policy, so that we conform to the simulate API.
    temp_dir = tempfile.TemporaryDirectory()
    policy_path = osp.join(temp_dir.name, "policy.pkl")
    torch.save(policy, policy_path, pickle_module=dill)

    game = empirical_game.init_game(saved_env_name=FLAGS.env)

    for opponent_i, opponent_path in enumerate(opponent_paths):
        if is_attacker:
            reward, _ = simulate_profile(
                env=game.env,
                game=game,
                nn_att=policy_path,
                nn_def=opponent_path,
                n_episodes=game.num_episodes,
                n_processes=1,
                raw_rewards=True)

        else:
            _, reward = simulate_profile(
                env=game.env,
                game=game,
                nn_att=opponent_path,
                nn_def=policy_path,
                n_episodes=game.num_episodes,
                n_processes=1,
                raw_rewards=True)

        expected_reward += mixture[opponent_i] * np.mean(reward)
        expected_reward_std += (mixture[opponent_i]**2) * (np.std(reward)**2)

    return expected_reward, np.sqrt(expected_reward_std)


def merge_replay_buffers(replay_buffers):
    """ Merge several replay buffers. """
    if len(replay_buffers) < 1:
        raise ValueError("Must provide replay buffers.")
    if len(replay_buffers) == 1:
        return copy.deepcopy(replay_buffers[0])

    # Create a destination buffer that will contain the union of all experiences.
    total_buffer_size = sum([len(b) for b in replay_buffers])
    merged_buffer = ReplayBuffer(size=total_buffer_size)

    # Put data from all buffers into single merged buffer.
    for buffer in replay_buffers:
        for datum in buffer._storage:
            merged_buffer.add(*datum)

    return merged_buffer
