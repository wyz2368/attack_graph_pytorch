""" Main training script. """
import logging
import multiprocessing
import os
import os.path as osp
import warnings

import gin
import numpy as np
import torch
from absl import app, flags

import attackgraph.settings as settings
from attackgraph import double_oracle

# Command line flags.
flags.DEFINE_string(
    "env",
    "10n11e",
    "Environment's name.")
flags.DEFINE_string(
    "run_name",
    None,
    "Experiment's run name.")
flags.DEFINE_string(
    "meta_method",
    "nash",
    "EGTA algorithm.")
flags.DEFINE_multi_string(
    "config_files",
    None,
    "Name of the gin config files to use.")
flags.DEFINE_multi_string(
    "config_overrides",
    [],
    "Overrides for gin config values.")
FLAGS = flags.FLAGS


def main(argv):
    """ Run training script. """
    # Configure information displayed to terminal.
    np.set_printoptions(precision=2)
    warnings.filterwarnings("ignore")

    # Set-up the result directory.
    run_dir = osp.join(settings.SRC_DIR, "results", FLAGS.run_name)
    if osp.exists(run_dir):
        print("Cannot resume previously saved run, overwriting data.")
    else:
        os.mkdir(run_dir)

        sub_dirs = [
            "attacker_policies",
            "attacker_best_responses",
            "defender_policies",
            "defender_best_responses"]
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

    # Run EGTA.
    double_oracle.run(load_env=FLAGS.env, env_name=None, n_processes=gin.REQUIRED)



if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    flags.mark_flag_as_required("run_name")
    app.run(main)
