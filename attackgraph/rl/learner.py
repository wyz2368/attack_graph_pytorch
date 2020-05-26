""" Training loop for a pytorch DQN agent. """
import os.path as osp
import tempfile
import time
import typing
from dataclasses import dataclass

import dill
import gin
import numpy as np
import torch

import attackgraph.common.file_ops as fp
from attackgraph import settings
from attackgraph.opponent_sampler import OpponentSampler
from attackgraph.rl.modules.replay_buffer import ReplayBuffer
from attackgraph.rl.modules.schedules import LinearSchedule
from attackgraph.util import mask_generator_att


def get_new_policy(locals_: dict = {}, globals_: dict = {}):
    """ Returns a new instance of the policy to train. """
    raise NotImplementedError()


@gin.configurable
@dataclass
class Learner(object):

    get_new_policy: typing.Any = get_new_policy

    seed: int = None
    total_timesteps: int = 100000
    buffer_size: int = 30000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.02
    train_freq: int = 1
    batch_size: int = 32
    print_freq: int = 100
    checkpoint_freq: int = 10000
    checkpoint_path: str = None
    learning_starts: int = 1000
    gamma: float = 0.99
    target_network_update_freq: int = 500
    prioritized_replay: bool = False
    param_noise: bool = False
    callback: typing.Any = None
    load_path: str = None
    progress_bar: bool = False
    reward_shaping: bool = True

    @gin.configurable
    def learn_multi_nets(self, env, epoch, writer, **network_kwargs):
        """ Train a deepq model.

        :param env: Environment.
        :param epoch: Current EGTA epoch. This is only used for saving results.
        :param writer: SummaryWriter for logging metrics.
        """
        time_init = time.time()
        # If the training flag is 1 we're training the attacker, or the defender if the flag is 0.
        training_attacker = env.training_flag
        assert training_attacker == 0 or training_attacker == 1, f"Invalid training flag: {training_attacker}."

        log_prefix = "attacker" if training_attacker else "defender"

        # Select parameters based off attacker/defender.
        n_actions = env.act_dim_att() if training_attacker else env.act_dim_def()
        observation_space = env.obs_dim_att() if training_attacker else env.obs_dim_def()

        # Create the new agent that we are going to train to best respond.
        best_responder = self.get_new_policy(locals_=locals(), globals_=globals())

        # Set-up experience replay buffer.
        replay_buffer = ReplayBuffer(self.buffer_size)
        assert not self.prioritized_replay, "Prioirized replay is not implemented in PyTorch recreation."

        # Create exploration schedule.
        exploration = LinearSchedule(
            schedule_timesteps=int(self.exploration_fraction * self.total_timesteps),
            initial_p=self.exploration_initial_eps,
            final_p=self.exploration_final_eps)

        # Set-up training variables.
        mean_rewards = []
        episode_rewards = [0.0]
        saved_mean_reward = None

        # Begin episode.
        obs = env.reset_everything_with_return()
        reset = True

        # Sample our initial opponent's strategy.
        opponent_sampler = OpponentSampler(
            env=env,
            opponent_identity=0 if training_attacker else 1)
        opponent_sampler.sample()

        # Establish temporary directory to hold checkpoints of our agent from throughout training.
        # We do this so we can return the best version of our agent throughout training.
        temp_dir = tempfile.TemporaryDirectory()
        best_model_path = osp.join(temp_dir.name, "model.pytorch")

        # Time metrics.
        time_init = time.time() - time_init
        t_transitions = []
        t_actions = []
        t_steps = []
        t_samples = []
        t_updates = []
        n_updates = 0.0

        # Reward Shaping
        temp_buffer = []

        # Environment training loop.
        time_training = time.time()
        for t in range(self.total_timesteps):
            time_transition = time.time()

            # Check terminantion conditions.
            if self.callback is not None and self.callback(locals(), globals()):
                break

            # Collect meta-data agent may need to compute action.
            time_action = time.time()
            action_kwargs = {}

            # Update exploration strategy.
            if self.param_noise:
                update_eps = 0.0
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps=exploration.value(t).
                # See Appendix C.1 in `Parameter Space Noise for Exploration`, Plappert et al., 2017.
                update_param_noise_threshold = -1.0 * np.log(
                    1.0 - exploration.value(t) + exploration.value(t)/float(env.action_space.n))
                action_kwargs["reset"] = reset
                action_kwargs["update_param_noise_threshold"] = update_param_noise_threshold
                action_kwargs["update_param_noise_scale"] = True

            else:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.0

            # If we are the attacker, apply a mask to our action space.
            if training_attacker:
                mask = mask_generator_att(env, np.array(obs)[None])
            else:
                mask = None

            # Step agent.
            writer.add_scalar(f"{log_prefix}/epsilon", update_eps, t)
            action = best_responder.act(
                observation=np.array(obs)[None],
                stochastic=True,
                update_eps=update_eps,
                mask=mask,
                training_attacker=training_attacker,
                **action_kwargs)[0]
            t_actions += [time.time() - time_action]

            # Step environment.
            time_step = time.time()
            new_obs, reward, done = env.step(action)
            t_steps += [time.time() - time_step]

            # Store transition data.
            # Reward shaping
            if self.reward_shaping:
                pass_flag = False
                if training_attacker == 0:
                    rewards_shaping = env.rewards()
                    if rewards_shaping['pass_flag']:
                        for transition in temp_buffer:
                            obs0, action0, rew0, new_obs0, done0 = transition
                            rew_new = rewards_shaping[str(action0)].v
                            episode_rewards[-1] += rew_new
                            replay_buffer.add(obs0, action0, rew_new, new_obs0, done0)
                        temp_buffer = []
                        env.reset_reward_shaping()
                        pass_flag = True
                elif training_attacker == 1:
                    rewards_shaping = env.rewards()
                    if rewards_shaping['pass_flag']:
                        for transition in temp_buffer:
                            obs1, action1, rew1, new_obs1, done1 = transition
                            rew_new = rewards_shaping[str(action1)].v
                            episode_rewards[-1] += rew_new
                            replay_buffer.add(obs1, action1, rew_new, new_obs1, done1)
                        temp_buffer = []
                        env.reset_reward_shaping()
                        pass_flag = True

                if pass_flag:
                    episode_rewards[-1] += reward
                    replay_buffer.add(obs, action, reward, new_obs, float(done))
                else:
                    temp_buffer.append((obs, action, reward, new_obs, float(done)))

                obs = new_obs

                if done:
                    obs = env.reset_everything_with_return()
                    episode_rewards.append(0.0)
                    reset = True
                    # sample a new strategy from meta-stategy solver.
                    opponent_sampler.sample()

            # No reward shaping.
            else:
                replay_buffer.add(obs, action, reward, new_obs, float(done))
                obs = new_obs
                episode_rewards[-1] += reward

                # If the environment finished, reset the environment and sample from opponent's meta-strategy.
                if done:
                    obs = env.reset_everything_with_return()
                    opponent_sampler.sample()

                    # Log the environment reset.
                    episode_rewards.append(0.0)
                    reset = True

            # Periodically train our policy.
            if (t > self.learning_starts) and (t % self.train_freq == 0):
                n_updates += 1.0
                time_sample = time.time()
                # Collect batch (b) of experiences.
                b_o, b_a, b_r, b_op, b_d = replay_buffer.sample(self.batch_size)
                b_weights = np.ones_like(b_r)

                # Generate action masks.
                if training_attacker:
                    b_mask = mask_generator_att(env, b_op)
                else:
                    b_mask = None

                t_samples += [time.time() - time_sample]

                time_update = time.time()
                best_responder.update(
                    observations=b_o,
                    actions=b_a,
                    rewards=b_r,
                    next_observations=b_op,
                    done_mask=b_d,
                    importance_weights=b_weights,
                    mask=b_mask,
                    training_attacker=training_attacker,
                    summary_writer=writer,
                    t=t)
                t_updates += [time.time() - time_update]

            # Periodically update target network.
            if (t > self.learning_starts) and (t % self.target_network_update_freq == 0):
                best_responder.update_target_network()

            # Record results.
            n_episodes = len(episode_rewards)
            if t > self.learning_starts:
                mean_100ep_reward = round(np.mean(episode_rewards[-251:-1]), 1)
                mean_rewards.append(mean_100ep_reward)
                writer.add_scalar(f"{log_prefix}/mean_reward", np.nan_to_num(mean_100ep_reward), t)

            # Periodically save a snapshot of our best-responder.
            if (self.checkpoint_freq is not None) and (t > self.learning_starts) and (n_episodes > 100) and (t % self.checkpoint_freq == 0):
                # Save checkpoints of only the best-performing model we have encountered.
                if (saved_mean_reward is None) or (mean_100ep_reward > saved_mean_reward):
                    torch.save(best_responder, best_model_path, pickle_module=dill)
                    saved_mean_reward = mean_100ep_reward

            t_transitions += [time.time() - time_transition]

        # Load the best-performing encountered policy as our resulting best-responder.
        BD = None
        if osp.exists(best_model_path):
            best_responder = torch.load(best_model_path)
            BD = saved_mean_reward if saved_mean_reward is not None else mean_100ep_reward

        # Clean-up temporary directory.
        temp_dir.cleanup()

        # Save data to generate learning curves.
        name = "attacker" if training_attacker else "defender"
        data_path = osp.join(settings.get_run_dir(), f"mean_rewards.{name}.{epoch}.pkl")
        fp.save_pkl(mean_rewards, data_path)

        # Log timing statistics.
        # We put this together into a string to send back to have the main process print.
        # This is to prevent potential multiprocessing errors.
        report = ""
        report += "  - n_transitions: {}\n".format(len(t_transitions))
        report += "  - n_updates: {}\n".format(len(t_updates))
        report += "  - t_init: {}\n".format(time_init)
        report += "  - t_transitions: {}\n".format(np.mean(t_transitions))
        report += "  - t_actions: {}\n".format(np.mean(t_actions))
        report += "  - t_steps: {}\n".format(np.mean(t_steps))
        report += "  - t_samples: {}\n".format(np.mean(t_samples))
        report += "  - t_updates: {}\n".format(np.mean(t_updates))

        return best_responder, BD, replay_buffer, report
