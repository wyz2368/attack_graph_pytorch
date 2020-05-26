""" Perform policy distillations on a QMix policy. """
import logging
import multiprocessing
import os
import os.path as osp
import shutil
import sys
import tempfile
import typing
import warnings

import dill
import gin
import numpy as np
import torch
from absl import app, flags
from tensorboardX import SummaryWriter

import attackgraph.common.file_ops as fp
from attackgraph import empirical_game, settings, training
from attackgraph.policy_distillation import distill_policy, merge_replay_buffers, simulate_rewards
from attackgraph.rl.dqn.dqn import DQN
from attackgraph.rl.modules import LazyLoadNN, QMixture
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

    game = empirical_game.init_game(saved_env_name=FLAGS.env)
    student = _policy_distillation(
        teacher_path=gin.REQUIRED,
        student_ctor=DQN,
        target_path=gin.REQUIRED,
        buffer_paths=gin.REQUIRED,
        env=game.env,
        training_attacker=gin.REQUIRED)


@gin.configurable
def _policy_distillation(teacher_path, student_ctor, target_path, buffer_paths, env, training_attacker: bool):
    """ Entry point for running policy distilation during the Mixed Oracle algorthm.

    :param teacher_path: Path to the teacher policy.
    :param student_ctor: Constructor for the student policy. Requires `state_dim` and `action_dim` input.
    :param target_path: Path to the target policy of the student. In the case of QMixing, this is the
      policy trained against the mixture opponent.
    :param env: Environment.
    :training_attacker: Boolean for determining if we are training an attacker policy.
    :return:
    """
    # Build a student, via looking up the correct environment dimensions.
    state_dim = env.obs_dim_att() if training_attacker else env.obs_dim_def()
    action_dim = env.act_dim_att() if training_attacker else env.act_dim_def()
    student = student_ctor(
        is_attacker=training_attacker,
        input_size=state_dim,
        hidden_sizes=gin.REQUIRED,
        output_size=action_dim)

    temp_dir = tempfile.TemporaryDirectory()
    policy_path = osp.join(temp_dir.name, "policy.pkl")
    torch.save(student, policy_path, pickle_module=dill)

    logger.info(f"Loading teacher from: {teacher_path}")
    teacher = torch.load(teacher_path)
    logger.info(f"Teacher loaded: {teacher}")
    logger.info(f"Loading target from: {target_path}")
    target = torch.load(target_path)
    logger.info(f"Target loaded: {target}")

    # Load all the replay buffers and merge/split them.
    logger.info(f"Loading replay buffers from: ")
    replay_buffers = []
    for path in buffer_paths:
        logger.info(f"  - {path}")
        replay_buffers += [fp.load_pkl(path)]
    replay_buffer = merge_replay_buffers(replay_buffers)

    # Calculate baseline performance.
    logger.info("Simulate target-policy and teacher-policy as upper-bounds on performance.")
    reward_mean, reward_std = simulate_rewards(policy=teacher)
    logger.info(f"  - Teacher reward: {reward_mean}, {reward_std}")
    reward_mean, reward_std = simulate_rewards(policy=target)
    logger.info(f"  - Target reward: {reward_mean}, {reward_std}")

    # Run policy distillation.
    logger.info("Running policy distillation.")
    student = distill_policy(
        teacher=teacher,
        student=student,
        env=env,
        dataset=replay_buffer,
        training_attacker=training_attacker)
    return student


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    app.run(main)
