""" Evaluate the quality of a learning algorithm by training it against a static set of opponents.

The goal of this script is to provide some way to compare learning algorithms in a systematic way. In order to
accomplish this we will have to define:
 (1) the environment, and
 (2) the other agents in the environment;
so we have some static training benchmark. Currently we cannot compare two EGTA runs because they are learning
best response policies to independent sets of opponent strategies. Therefore we cannot tell if a certain set
of hyperparameters is actually better than another ste of hyperparmeters.

This script works by evaluating a single learning algorithm (e.g., set of hyperparameters) against an
environment and a set of opponents. The learning algorithm will learn a best-response against each
opponent policy independnetly, and we will average the resulting metrics. Then we can direclty compare
the quality of each respective learning algorithm.

The opponents are all taken from data/evaluation_opponents/(attacker|defender).
"""
import glob
import logging
import multiprocessing
import os
import os.path as osp
import shutil
import sys
import warnings

import gin
import numpy as np
import torch
from absl import app, flags
from tensorboardX import SummaryWriter

import attackgraph.common.file_ops as fp
from attackgraph import empirical_game, settings, training
from attackgraph.common.cloudpickle_wrapper import CloudpickleWrapper

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

logger = logging.getLogger(__name__)


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

    evaluate_algorithm()


@gin.configurable
def evaluate_algorithm(train_attacker: bool, n_processes: int = 1):
    """ Evaluate learning algorithm.

    :param train_attacker: Evaluate learning an attacker policy.
    :type train_attacker: bool
    :param n_processes: Number of processors available.
    :type n_processes: int
    """
    # Load the paths to the opponent policies.
    data_dir = osp.join(settings.SRC_DIR, "data", "evaluation_opponents")
    data_dir = osp.join(data_dir, "defenders" if train_attacker else "attackers")

    opponent_policy_paths = glob.glob(osp.join(data_dir, "*"))
    name = "attacker" if train_attacker else "defender"
    logger.info(f"Training seperate {name} policies against the following opponent(s):")
    for opponent in opponent_policy_paths:
        logger.info(f"  - {opponent}")

    game = empirical_game.init_game(saved_env_name=FLAGS.env)

    # Set-up computational resources to distribute jobs across.
    logger.info(f"Distributing learning across {n_processes} proccessor(s).")
    job_queue = multiprocessing.SimpleQueue()
    workers = []
    for i in range(n_processes):
        workers += [EvaluationWorker(
            id=i,
            job_queue=job_queue,
            train_attacker=train_attacker)]
        workers[-1].start()

    # Submit evaluation jobs to the queue.
    for opponent in opponent_policy_paths:
        job_queue.put((CloudpickleWrapper(game), opponent))
    # Send sentinel values to tell processes to cleanly shutdown (1 per worker).
    for _ in range(n_processes):
        job_queue.put(None)
    for worker in workers:
        worker.join()


class EvaluationWorker(multiprocessing.Process):

    def __init__(self, id: int, job_queue: multiprocessing.SimpleQueue, train_attacker: bool):
        super(EvaluationWorker, self).__init__()
        self.id = id
        self.job_queue = job_queue
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

        policy_name = "attacker" if self.train_attacker else "defender"

        for job in iter(self.job_queue.get, None):
            # The game we're given has no policies and has not been initialized.
            game, opponent = job
            game = game()  # Unpickle game.

            # Register the opponent we will be playing as the opponent's only policy.
            if self.train_attacker:
                game.add_def_str(opponent)
            else:
                game.add_att_str(opponent)

            # The opponent sampling is done from the result directory, so we need
            # to copy any model we use into the policy set.
            if self.train_attacker:
                opponent_dir = settings.get_defender_strategy_dir()
            else:
                opponent_dir = settings.get_attacker_strategy_dir()
            new_filepath = osp.join(opponent_dir, osp.basename(opponent))
            shutil.copyfile(src=opponent, dst=new_filepath)

            save_path = osp.join(settings.get_run_dir(), osp.basename(opponent))
            save_path = save_path[:-4]  # Remove ".pkl".
            training.train(
                game=game,
                identity=int(self.train_attacker),
                opponent_mix_str=np.array([1.0]),
                epoch=osp.basename(opponent),
                writer=SummaryWriter(logdir=save_path),
                save_path=osp.join(save_path, f"{policy_name}.pkl"))


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    app.run(main)
