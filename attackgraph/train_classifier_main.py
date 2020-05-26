""" Train an opponent classifier. """
import logging
import multiprocessing
import os
import os.path as osp
import shutil
import sys
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
from attackgraph.policy_distillation import distill_policy, merge_replay_buffers
from attackgraph.rl.modules import LazyLoadNN, QMixture
from attackgraph.rl.modules.mlp import MLP
from attackgraph.simulation import simulate_profile
from attackgraph.supervised_learning import supervised_learning


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
    student = _train_classifier(
        classifier=MLP(),
        buffer_paths=gin.REQUIRED,
        mixture=gin.REQUIRED,
        env=game.env,
        training_attacker=gin.REQUIRED)


@gin.configurable
def _train_classifier(classifier, buffer_paths, mixture, env, test_split: float, training_attacker: bool):
    """ Train an opponent classifier. """
    # Load all the replay buffers and merge/split them.
    logger.info(f"Loading replay buffers from: ")
    labels = []
    replay_buffers = []
    for buffer_i, path in enumerate(buffer_paths):
        logger.info(f"  - {path}")
        replay_buffers += [fp.load_pkl(path)]
        labels += [np.ones([len(replay_buffers[-1])]) * buffer_i]
    replay_buffer = merge_replay_buffers(replay_buffers)
    # We only want the state.
    replay_buffer = [x[0] for x in replay_buffer._storage]
    replay_buffer = np.array(replay_buffer)
    labels = np.ravel(labels)

    assert replay_buffer.shape[0] == labels.shape[0]

    # Shuffle the data.
    new_indices = np.random.permutation(len(labels))
    replay_buffer = replay_buffer[new_indices]
    labels = labels[new_indices]

    # Train/test split.
    n_test_data = int(len(labels)*test_split)

    # Train the opponent classifier.
    classifier = supervised_learning(
        net=classifier,
        train_X=replay_buffer[:-n_test_data],
        train_Y=labels[:-n_test_data],
        test_X=replay_buffer[-n_test_data:],
        test_Y=labels[-n_test_data:],
        criterion=gin.REQUIRED,
        n_epochs=gin.REQUIRED,
        eval_freq=gin.REQUIRED,
        batch_size=gin.REQUIRED,
        log_dir=settings.get_run_dir())
    return student


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    app.run(main)
