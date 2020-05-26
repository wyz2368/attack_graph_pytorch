""" Gambit utility functions.

Input arguments: payoff matrix for the defender, poDef; payoff matrix for the attacker, poAtt.
In a payoff matrix example: 1 2
                              3 5
                              6 7
There are 3 defender strategies (3 rows) and 2 attacker strategies (2 columns).
NFG File format: Payoff version
"""
import logging
import os.path as osp

import numpy as np
from absl import flags

import attackgraph.common.file_ops as fp
from attackgraph import settings, subproc

FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)


def read_normal_form_game_file(path: str):
    """ Parses a normal form game file (*.nfg).

    Resources:
     - gambit-project.org/gambit13/formats.html

    :param path: Filepath (*.nfg).
    :type path: str
    :return: The payout matrices for the Defender then Attacker.
    :rtype: (np.ndarray, np.ndarray)
    """
    if not osp.exists(path):
        raise ValueError(f"File does not exist: {path}.")
    if not path.endswith(".nfg"):
        raise ValueError(f"Expected *.nfg filetype: {path}.")

    with open(path, 'r') as gambit_file:
        # Parse file prologue.
        _ = gambit_file.readline()
        # Parse player and game meta data.
        strategy_labels = gambit_file.readline().split(" ")
        _, row_player, col_player, _, _, n_rows, n_cols, _ = strategy_labels
        n_rows = int(n_rows)
        n_cols = int(n_cols)
        # Skip empty line.
        gambit_file.readline()
        # Parse pay-off matrix.
        payout_def = np.zeros([n_rows, n_cols])
        payout_att = np.zeros([n_rows, n_cols])
        payouts = gambit_file.readline().strip().split(" ")
        payout_index = 0

        for col in range(n_cols):
            for row in range(n_rows):
                payout_def[row][col] = float(payouts[payout_index])
                payout_att[row][col] = float(payouts[payout_index + 1])
                payout_index += 2

    return payout_def, payout_att


def _get_gambit_matrix_path():
    """ Get path to Gambit's payoff matrix.

    :return: Filepath.
    :rtype: str
    """
    gambit_path = osp.join(settings.get_run_dir(), "payoff_matrix.nfg")
    return gambit_path


def _get_gambit_nash_path():
    """ Get path to Gambit's payoff matrix.

    :return: Filepath.
    :rtype: str
    """
    gambit_path = osp.join(settings.get_run_dir(), "nash.txt")
    return gambit_path


def gambit_analysis(timeout, in_path: str = None, out_path: str = None):
    """ Run gambit analysis on the current empirical game.

    :param timeout:
    :type timeout: int
    """
    in_path = in_path or _get_gambit_matrix_path()
    out_path = out_path or _get_gambit_nash_path()

    if not fp.isExist(in_path):
        raise ValueError(".nfg file does not exist!")
    command_str = "gambit-lcp "
    command_str += "-q {} ".format(in_path)
    command_str += "-d 8 "
    command_str += "> {}".format(out_path)
    subproc.call_and_wait_with_timeout(command_str, timeout)


def encode_gambit_file(poDef, poAtt, path: str = None):
    """ Encode pay-off matrices into gambit's file format.

    :param poDef:
    :type poDef:
    :param poAtt:
    :type poAtt:
    """
    path = path or _get_gambit_matrix_path()

    try:
        if poDef.shape != poAtt.shape:
            raise Exception("Inputted payoff matrix for defender and attacker must be of same shape.")
    except Exception as error:
        logger.info(repr(error))
        return -1
    # Write header
    with open(path, "w") as nfgFile:
        nfgFile.write('NFG 1 R "Attackgroup"\n{ "Defender" "Attacker" } ')
        # Write strategies
        nfgFile.write('{ ' + str(poDef.shape[0]) + ' ' + str(poDef.shape[1]) + ' }\n\n')
        # Write outcomes
        for i in range(poDef.shape[1]):
            for j in range(poDef.shape[0]):
                nfgFile.write(str(poDef[j][i]) + " ")
                nfgFile.write(str(poAtt[j][i]) + " ")

    # Gambit passing and NE calculation to come later.


def decode_gambit_file(path: str = None):
    path = path or _get_gambit_nash_path()

    if not fp.isExist(path):
        raise ValueError("nash.txt file does not exist!")
    with open(path, 'r') as f:
        nash = f.readline()
        if len(nash.strip()) == 0:
            return 0, 0

    nash = nash[3:]
    nash = nash.split(',')
    new_nash = []
    for i in range(len(nash)):
        new_nash.append(convert(nash[i]))

    new_nash = np.array(new_nash)
    new_nash = np.round(new_nash, decimals=8)
    nash_def = new_nash[:int(len(new_nash)/2)]
    nash_att = new_nash[int(len(new_nash)/2):]

    return nash_att, nash_def


def do_gambit_analysis(poDef, poAtt):
    timeout = 600
    encode_gambit_file(poDef, poAtt)  # TODO:change timeout adaptive
    while True:
        gambit_analysis(timeout)
        nash_att, nash_def = decode_gambit_file()
        timeout += 120
        if timeout > 7200:
            logger.critical("Gambit has been running for more than 2 hour.!")
        if isinstance(nash_def, np.ndarray) and isinstance(nash_att, np.ndarray):
            break
        logger.warning("Timeout has been added by 120s.")
    logger.info('gambit_analysis done!')
    return nash_att, nash_def


def convert(s):
    try:
        return float(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)


# ne is a dic. nash is a numpy. 0: def, 1: att
def add_new_NE(game, nash_att, nash_def, epoch):
    if not isinstance(nash_att, np.ndarray):
        raise ValueError("nash_att is not numpy array.")
    if not isinstance(nash_def, np.ndarray):
        raise ValueError("nash_def is not numpy array.")
    if not isinstance(epoch, int):
        raise ValueError("Epoch is not an integer.")
    ne = {}
    ne[0] = nash_def
    ne[1] = nash_att
    game.add_nasheq(epoch, ne)
