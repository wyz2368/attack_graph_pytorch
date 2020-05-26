""" Global module settings. """
import os.path as osp

from absl import flags

FLAGS = flags.FLAGS


# Source code directory.
SRC_DIR = osp.dirname(osp.abspath(__file__))
SRC_DIR = osp.join(SRC_DIR, "..")


def get_results_dir():
    """ Get the run's result directory.

    :return: Run's result directory.
    :rtype: str
    """
    results_dir = osp.join(SRC_DIR, "results")
    return results_dir


def get_run_dir():
    """ Get the run's result directory.

    :return: Run's result directory.
    :rtype: str
    """
    run_dir = osp.join(get_results_dir(), FLAGS.run_name)
    return run_dir


def get_strategy_dir(name: str):
    """

    :param name:
    :type name: str
    :return: Directory path.
    :rtype: str
    """
    assert name in ["attacker", "defender"]
    run_dir = get_run_dir()
    strategy_dir = osp.join(run_dir, "{}_policies".format(name))
    return strategy_dir


def get_attacker_strategy_dir():
    return get_strategy_dir("attacker")


def get_defender_strategy_dir():
    return get_strategy_dir("defender")


def get_best_response_dir(name: str):
    assert name in ["attacker", "defender"]
    run_dir = get_run_dir()
    strategy_dir = osp.join(run_dir, "{}_best_responses".format(name))
    return strategy_dir


def get_attacker_best_response_dir():
    return get_best_response_dir("attacker")


def get_defender_best_response_dir():
    return get_best_response_dir("defender")


def get_env_data_dir():
    return osp.join(SRC_DIR, "data", "env")
