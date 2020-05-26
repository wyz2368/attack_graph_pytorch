""" Evaluate the quality of a QMixture compared to learned best response. """
import logging
import multiprocessing
import os
import os.path as osp
import shutil
import sys
import typing
import warnings
from collections import defaultdict

import dill
import gin
import numpy as np
import torch
from absl import app, flags
from tensorboardX import SummaryWriter

from attackgraph import empirical_game, settings, training
from attackgraph.double_oracle import initialize
from attackgraph.policy_distillation import merge_replay_buffers
from attackgraph.rl.modules import LazyLoadNN, QMixture, QMixtureStateFreq, QMixtureSubStateFreq
from attackgraph.simulation import simulate_profile

# Command line flags.
flags.DEFINE_string(
    "env",
    "run_env_B",
    "Environment's name 1.")
flags.DEFINE_string(
    "run_name",
    None,
    "Experiment's run name.")
flags.DEFINE_multi_string(
    "config_files",
    None,
    "Name of the gin config files to use.")
flags.DEFINE_multi_string(
    "config_overrides",
    [],
    "Overrides for gin config values.")
FLAGS = flags.FLAGS
flags.mark_flag_as_required("run_name")

logger = logging.getLogger("attackgraph")


def main(argv):
    """ Run evaluation script.

    :param argv: Command line arguments.
    """
    # Configure information displayed to terminal.
    np.set_printoptions(precision=2)
    warnings.filterwarnings("ignore")

    # Set-up the result directory.
    run_dir = settings.get_run_dir()
    if osp.exists(run_dir):
        print("Cannot resume previously saved run, overwriting data.")
    else:
        os.mkdir(run_dir)

        sub_dirs = [
            "attacker_policies",
            "defender_policies"]
        for sub_dir in sub_dirs:
            sub_dir = osp.join(run_dir, sub_dir)
            os.mkdir(sub_dir)

    # Set-up logging.
    logger = logging.getLogger("attackgraph")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []  # absl has a default handler that we need to remove.
    # logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    # Log to terminal.
    terminal_handler = logging.StreamHandler()
    terminal_handler.setFormatter(formatter)
    # Log to file.
    file_handler = logging.FileHandler(osp.join(run_dir, "out.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    # Debug output.
    debug_handler = logging.FileHandler(osp.join(run_dir, "debug.log"))
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    # Register handlers.
    logger.addHandler(terminal_handler)

    logger.addHandler(file_handler)
    logger.addHandler(debug_handler)

    logger.info(f"Saving results to: {run_dir}")

    # Set-up gin configuration.
    gin_files = [osp.join(settings.SRC_DIR, "configs", f"{x}.gin") for x in FLAGS.config_files]
    gin.parse_config_files_and_bindings(
        config_files=gin_files,
        bindings=FLAGS.config_overrides,
        skip_unknown=False)

    # Save program flags.
    with open(osp.join(run_dir, "flags.txt"), "w") as flag_file:
        # We want only flags relevant to this module to be saved, no extra flags.
        # See: https://github.com/abseil/abseil-py/issues/92
        key_flags = FLAGS.get_key_flags_for_module(argv[0])
        key_flags = "\n".join(flag.serialize() for flag in key_flags)
        flag_file.write(key_flags)
    with open(osp.join(run_dir, "config.txt"), "w") as config_file:
        config_file.write(gin.config_str())

    # Properly restrict pytorch to not consume extra resources.
    #  - https://github.com/pytorch/pytorch/issues/975
    #  - https://github.com/ray-project/ray/issues/3609
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    evaluate_qmix()


@gin.configurable
def evaluate_qmix(train_attacker: bool, opponent_paths: typing.List, mixture: typing.List, n_processes: int = 1):
    """ . """
    assert len(opponent_paths) == len(mixture)

    game = empirical_game.init_game(saved_env_name=FLAGS.env)

    name = "attacker" if train_attacker else "defender"
    logger.info(f"Training best response {name} policies against the following opponent(s):")
    for path in opponent_paths:
        logger.info(f"  - {path}")

    # -------------------------------------------------------------------------
    # Train best-response to each opponent.
    logger.info("Training best-response against each pure-strategy.")
    job_queue = multiprocessing.SimpleQueue()

    workers = []
    for i in range(n_processes):
        workers += [TrainingWorker(
            id=i,
            job_queue=job_queue,
            scope="pure",
            train_attacker=train_attacker)]
        workers[-1].start()

    # Submit jobs to learn best responses to each pure strategy.
    best_response_paths = []
    for opponent_path in opponent_paths:
        br_path = osp.join(settings.get_run_dir(), osp.basename(opponent_path)[:-4]+".best_response.pkl")
        best_response_paths += [br_path]

        job = (
            # Save best-response path.
            br_path,
            # List of the opponent's policies.
            [opponent_path],
            # Mixture of the opponent's policies.
            [1.0])
        job_queue.put(job)

    # Send end flag and join workers.
    for _ in range(n_processes):
        job_queue.put(None)
    for worker in workers:
        worker.join()

    # -------------------------------------------------------------------------
    # Simulate the performance of QMixture.
    logger.info("Simulating the performance of the QMixture.")
    q_funcs = [LazyLoadNN(save_path=path) for path in best_response_paths]
    qmix = QMixture(mixture=mixture, q_funcs=q_funcs)

    # Need to save qmix, so that the simulator can load the policy.
    qmix_path = osp.join(settings.get_run_dir(), "qmix.pkl")
    torch.save(qmix, qmix_path, pickle_module=dill)

    qmix_rewards = []
    mixed_reward = 0.0
    for opponent_i, opponent_path in enumerate(opponent_paths):
        if train_attacker:
            rewards, _ = simulate_profile(
                env=game.env,
                game=game,
                nn_att=qmix_path,
                nn_def=opponent_path,
                n_episodes=game.num_episodes,
                n_processes=1,
                save_dir=None,
                summary_writer=None,
                raw_rewards=True)
        else:
            _, rewards = simulate_profile(
                env=game.env,
                game=game,
                nn_att=opponent_path,
                nn_def=qmix_path,
                n_episodes=game.num_episodes,
                n_processes=1,
                save_dir=None,
                summary_writer=None,
                raw_rewards=True)

        logger.info(f"  - Opponent {opponent_i}: {np.mean(rewards)}, {np.std(rewards)}")
        with open(osp.join(settings.get_run_dir(), best_response_paths[opponent_i][:-4], f"qmix.rewards.pkl"), "wb") as outfile:
            dill.dump(rewards, outfile)
        qmix_rewards += [rewards]
        mixed_reward += mixture[opponent_i] * np.mean(rewards)
    logger.info(f"Expected reward against mixture opponent: {mixed_reward}")

    def _std(rewards_):
        total = 0.0
        for coeff, reward_ in zip(mixture, rewards_):
            total += coeff**2 * np.std(reward_)**2
        return np.sqrt(total)

    logger.info(f"Expected std against mixture opponent: {_std(qmix_rewards)}")

    dill.dump(mixed_reward, open(osp.join(settings.get_run_dir(), "qmix.simulated_reward.pkl"), "wb"))

    """
    # -------------------------------------------------------------------------
    # Simulate the performance of QMixture with state frequencies.
    logger.info("Simulating the performance of the QMixture with State-Frequency weighting.")
    q_funcs = [LazyLoadNN(save_path=path) for path in best_response_paths]
    # We need to load the replay buffers in order to have an approximate state-frequency measure.
    replay_buffers = []
    for path in best_response_paths:
        buffer_path = f"{path[:-4]}.replay_buffer.pkl"
        with open(buffer_path, "rb") as buffer_file:
            replay_buffers += [dill.load(buffer_file)]

    qmix_statefreq = QMixtureStateFreq(mixture=mixture, q_funcs=q_funcs, replay_buffers=replay_buffers)
    game = empirical_game.init_game(saved_env_name=FLAGS.env)

    # Need to save qmix, so that the simulator can load the policy.
    qmix_statefreq_path = osp.join(settings.get_run_dir(), "qmix_statefreq.pkl")
    torch.save(qmix_statefreq, qmix_statefreq_path, pickle_module=dill)

    qmix_statefreq_rewards = []
    mixed_statefreq_reward = 0.0
    for opponent_i, opponent_path in enumerate(opponent_paths):
        if train_attacker:
            rewards, _, trajectories = simulate_profile(
                env=game.env,
                game=game,
                nn_att=qmix_statefreq_path,
                nn_def=opponent_path,
                n_episodes=game.num_episodes,
                n_processes=1,
                save_dir=None,
                summary_writer=None,
                raw_rewards=True,
                collect_trajectories=True)
        else:
            _, rewards, trajectories = simulate_profile(
                env=game.env,
                game=game,
                nn_att=opponent_path,
                nn_def=qmix_statefreq_path,
                n_episodes=game.num_episodes,
                n_processes=1,
                save_dir=None,
                summary_writer=None,
                raw_rewards=True,
                collect_trajectories=True)

        logger.info(f"  - Opponent {opponent_i}: {np.mean(rewards)}, {np.std(rewards)}")
        with open(osp.join(settings.get_run_dir(), best_response_paths[opponent_i][:-4], f"qmix_statefreq.rewards.pkl"), "wb") as outfile:
            dill.dump(rewards, outfile)
        qmix_statefreq_rewards += [rewards]
        mixed_statefreq_reward += mixture[opponent_i] * np.mean(rewards)
    logger.info(f"Expected reward against mixture opponent: {mixed_statefreq_reward}")
    logger.info(f"Expected std against mixture opponent: {_std(qmix_statefreq_rewards)}")
    dill.dump(mixed_reward, open(osp.join(settings.get_run_dir(), "qmix_statefreq.simulated_reward.pkl"), "wb"))
    dill.dump(trajectories, open(osp.join(settings.get_run_dir(), "qmix_statefreq.trajectories.pkl"), "wb"))

    # -------------------------------------------------------------------------
    logger.info("Inspecting trajectories generated to analyze State-Freq results.")

    def _in_qmix_buffers(o):
        found = False
        for buffer in qmix_statefreq.replay_buffers:
            for buffered_exp in buffer._storage:
                buffered_obs, *_ = buffered_exp

                if np.all(o[:90] == buffered_obs[:90]):
                    found = True
                    break
            if found:
                break
        return found

    # Collect the time-step (including action-set building), when we encounter a state not found in
    # the replay buffers.
    logger.info("Calculating the number of known sub-states encountered during simulation.")
    t_when_new_state = defaultdict(int)

    for traj in trajectories:
        t = 0
        done = False
        for exp in traj:
            # Get all of the observations for a single-timestep. This is multiple observations,
            # as a result of action-set building.
            observation = exp["observations"]["attacker"] if train_attacker else exp["observations"]["defender"]

            if _in_qmix_buffers(observation):
                t += 1
            else:
                t_when_new_state[t] += 1
                done = True
                break
        if done:
            break

    logger.info("t when new state is found: ")
    for count, freq in t_when_new_state.items():
        logger.info(f"  - t_{count}: {freq}")
    dill.dump(t_when_new_state, open(osp.join(settings.get_run_dir(), "qmix_statefreq.t_when_new_state.pkl"), "wb"))

    # Calculate the number of states that are known throughout the entire simulation.
    n_states = 0
    for traj in trajectories:
        for exp in traj:
            observations = exp["observations"]["attacker"] if train_attacker else exp["observations"]["defender"]
            n_states += len(observations)
    logger.info(f"  - Number of total states in simulation: {n_states}")

    n_known_states = 0
    for count, freq in t_when_new_state.items():
        n_known_states += count*freq
    logger.info(f"  - Number of known states: {n_known_states}")
    logger.info(f"  - % Known states: {n_known_states/float(n_states)}")

    # -------------------------------------------------------------------------
    # Simulate the performance of QMixture with state frequencies.
    logger.info("Simulating the performance of the QMixture with Sub-State-Frequency weighting.")
    q_funcs = [LazyLoadNN(save_path=path) for path in best_response_paths]
    # We need to load the replay buffers in order to have an approximate state-frequency measure.
    replay_buffers = []
    for path in best_response_paths:
        buffer_path = f"{path[:-4]}.replay_buffer.pkl"
        with open(buffer_path, "rb") as buffer_file:
            replay_buffers += [dill.load(buffer_file)]

    qmix_statefreq = QMixtureSubStateFreq(
        mixture=mixture,
        q_funcs=q_funcs,
        replay_buffers=replay_buffers,
        subspace_start=gin.REQUIRED,
        subspace_end=gin.REQUIRED)
    game = empirical_game.init_game(saved_env_name=FLAGS.env)

    # Need to save qmix, so that the simulator can load the policy.
    qmix_statefreq_path = osp.join(settings.get_run_dir(), "qmix_substate.pkl")
    torch.save(qmix_statefreq, qmix_statefreq_path, pickle_module=dill)

    qmix_statefreq_rewards = []
    mixed_statefreq_reward = 0.0
    for opponent_i, opponent_path in enumerate(opponent_paths):
        if train_attacker:
            rewards, _ = simulate_profile(
                env=game.env,
                game=game,
                nn_att=qmix_statefreq_path,
                nn_def=opponent_path,
                n_episodes=game.num_episodes,
                n_processes=1,
                save_dir=None,
                summary_writer=None,
                raw_rewards=True)
        else:
            _, rewards = simulate_profile(
                env=game.env,
                game=game,
                nn_att=opponent_path,
                nn_def=qmix_statefreq_path,
                n_episodes=game.num_episodes,
                n_processes=1,
                save_dir=None,
                summary_writer=None,
                raw_rewards=True)

        logger.info(f"  - Opponent {opponent_i}: {np.mean(rewards)}, {np.std(rewards)}")
        with open(osp.join(settings.get_run_dir(), best_response_paths[opponent_i][:-4], f"qmix_substate.rewards.pkl"), "wb") as outfile:
            dill.dump(rewards, outfile)
        qmix_statefreq_rewards += [rewards]
        mixed_statefreq_reward += mixture[opponent_i] * np.mean(rewards)
    logger.info(f"Expected reward against mixture opponent: {mixed_statefreq_reward}")
    logger.info(f"Expected std against mixture opponent: {_std(qmix_statefreq_rewards)}")
    dill.dump(mixed_reward, open(osp.join(settings.get_run_dir(), "qmix_substate.simulated_reward.pkl"), "wb"))
    """

    # -------------------------------------------------------------------------
    # Train best-response to opponent mixture.
    logger.info("Training a best-response against the mixture opponent.")
    mixture_br_path = osp.join(settings.get_run_dir(), "mixture.best_response.pkl")

    worker = TrainingWorker(
        id=i,
        job_queue=job_queue,
        scope="mix",
        train_attacker=train_attacker)
    worker.start()

    job_queue.put((
        # Save best-response path.
        mixture_br_path,
        # List of the opponent's policies.
        opponent_paths,
        # Mixture of the opponent's policies.
        mixture))
    job_queue.put(None)
    worker.join()

    # -------------------------------------------------------------------------
    # Evaluate the mixture policy against the individual opponent strategies.
    logger.info("Evaluating the best-response trained against mixture opponents on pure-strategy opponents.")

    mix_br_rewards = []
    mix_br_reward = 0.0
    for opponent_i, opponent_path in enumerate(opponent_paths):
        if train_attacker:
            rewards, _ = simulate_profile(
                env=game.env,
                game=game,
                nn_att=mixture_br_path,
                nn_def=opponent_path,
                n_episodes=game.num_episodes,
                n_processes=1,
                save_dir=None,
                summary_writer=None,
                raw_rewards=True)
        else:
            _, rewards = simulate_profile(
                env=game.env,
                game=game,
                nn_att=opponent_path,
                nn_def=mixture_br_path,
                n_episodes=game.num_episodes,
                n_processes=1,
                save_dir=None,
                summary_writer=None,
                raw_rewards=True)

        logger.info(f"  - Opponent {opponent_i}: {np.mean(rewards)}, {np.std(rewards)}")
        with open(osp.join(settings.get_run_dir(), best_response_paths[opponent_i][:-4], f"mixture_best_response.rewards.pkl"), "wb") as outfile:
            dill.dump(rewards, outfile)
        mix_br_reward += mixture[opponent_i] * np.mean(rewards)
        mix_br_rewards += [rewards]
    logger.info(f"Expected reward for mixture best-response: {mix_br_reward}")
    logger.info(f"Expected std against mixture opponent: {_std(mix_br_rewards)}")

    dill.dump(mix_br_reward, open(osp.join(settings.get_run_dir(), "mixture_best_response.simulated_reward.pkl"), "wb"))

    # -------------------------------------------------------------------------
    # Evaluate pure-strategy-best-response policies against all opponents (all pure strategy + mixture).
    logger.info("Evaluating pure-strategy-best-response against all opponent policies.")

    response_rewards = {}
    response_stds = {}
    for opponent_i, opponent_path in enumerate(opponent_paths):
        for response_i, best_response_path in enumerate(best_response_paths):
            if train_attacker:
                rewards, _ = simulate_profile(
                    env=game.env,
                    game=game,
                    nn_att=best_response_path,
                    nn_def=opponent_path,
                    n_episodes=game.num_episodes,
                    n_processes=1,
                    save_dir=None,
                    summary_writer=None,
                    raw_rewards=True)
            else:
                _, rewards = simulate_profile(
                    env=game.env,
                    game=game,
                    nn_att=opponent_path,
                    nn_def=best_response_path,
                    n_episodes=game.num_episodes,
                    n_processes=1,
                    save_dir=None,
                    summary_writer=None,
                    raw_rewards=True)

            logger.info(f"  - Opponent {opponent_i} vs. Best-Response {response_i}: {np.mean(rewards)}, {np.std(rewards)}")
            with open(osp.join(settings.get_run_dir(), f"opponent_{opponent_i}_v_response_{response_i}.rewards.pkl"), "wb") as outfile:
                dill.dump(rewards, outfile)

            if response_i not in response_rewards:
                response_rewards[response_i] = 0.0
                response_stds[response_i] = 0.0
            response_rewards[response_i] += mixture[opponent_i] * np.mean(rewards)
            response_stds[response_i] += mixture[response_i]**2 * np.std(rewards)**2

    for key, value in response_rewards.items():
        logger.info(f"Expected reward of response {key} against mixture: {value}")
    logger.info("Finished.")


class TrainingWorker(multiprocessing.Process):

    def __init__(self, id: int, job_queue: multiprocessing.SimpleQueue, scope: str, train_attacker: bool):
        super(TrainingWorker, self).__init__()
        self.id = id
        self.job_queue = job_queue
        self.scope = scope
        self.train_attacker = train_attacker

    def run(self):
        # Because we are "spawning" the process instead of "forking" the process, we need to
        # reimport the run's configurations.
        # Reparse the flags for this process.
        FLAGS = flags.FLAGS
        FLAGS(sys.argv)
        # Reload gin configurations for this process.
        gin_files = [osp.join(settings.SRC_DIR, "configs", f"{x}.gin") for x in FLAGS.config_files]
        gin.parse_config_files_and_bindings(
            config_files=gin_files,
            bindings=FLAGS.config_overrides,
            skip_unknown=False)

        for job in iter(self.job_queue.get, None):
            policy_save_path, opponents, mixture = job

            game = empirical_game.init_game(saved_env_name=FLAGS.env)

            # Register the opponents that we will be playing.
            opponent_dir = settings.get_defender_strategy_dir() if self.train_attacker else \
                settings.get_attacker_strategy_dir()
            for opponent_i, opponent_path in enumerate(opponents):
                # We add the opponent index to the end of the opponent name to deal
                # with any opponents with naming conflicts.
                opponent = f"{opponent_path[:-4]}_{opponent_i}_pid{self.id}.pkl"
                # Opponent sampling is done from the result directory, so we need
                # to copy any model we use into the policy set.
                new_filepath = osp.join(opponent_dir, osp.basename(opponent))
                shutil.copyfile(src=opponent_path, dst=new_filepath)
                if self.train_attacker:
                    game.add_def_str(new_filepath)
                else:
                    game.add_att_str(new_filepath)

            save_path = osp.join(settings.get_run_dir(), osp.basename(policy_save_path))
            save_path = save_path[:-4]  # Remove ".pkl".

            # Train a best-response.
            training.train(
                game=game,
                identity=int(self.train_attacker),
                opponent_mix_str=mixture,
                epoch=osp.basename(policy_save_path)[:-4],
                writer=SummaryWriter(logdir=save_path),
                save_path=policy_save_path,
                scope=self.scope)

            # Delete copied opponent policy.
            os.remove(new_filepath)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    app.run(main)
