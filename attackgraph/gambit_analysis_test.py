""" Test suite for `gambit_analysis`. """
import os.path as osp
import tempfile

import numpy as np
import pytest

import attackgraph.gambit_analysis as gambit_ops
from attackgraph import settings


def test_read_normal_form_game_file():
    """ Basic API test for `read_normal_form_game_file`. """
    game_path = osp.join(settings.SRC_DIR, "attackgraph", "test_data", "test_game.nfg")

    defender, attacker = gambit_ops.read_normal_form_game_file(game_path)

    np.testing.assert_array_equal(defender, [[1, 1], [0, 0], [0, 2]])
    np.testing.assert_array_equal(attacker, [[1, 1], [2, 3], [2, 0]])


def test_pure_equilibrium():
    """ Regression test for Gambit input/output. """
    po_def = np.array([[3, 0], [0, 2]])
    po_att = np.array([[2, 0], [0, 3]])

    temp_dir = tempfile.TemporaryDirectory()

    # Convert pay-off matrices into a normal form game file.
    gambit_input_path = osp.join(temp_dir.name, "game.nfg")
    gambit_ops.encode_gambit_file(po_def, po_att, gambit_input_path)

    # Run gambit analysis on the normal form game resulting in a Nash equilibrium file.
    gambit_output_path = osp.join(temp_dir.name, "nash.txt")
    gambit_ops.gambit_analysis(100, gambit_input_path, gambit_output_path)

    # Decode Nash file.
    nash_att, nash_def = gambit_ops.decode_gambit_file(gambit_output_path)
    np.testing.assert_array_equal(nash_att, [1, 0])
    np.testing.assert_array_equal(nash_def, [1, 0])

    temp_dir.cleanup()


def test_mixed_equilibrium():
    """ Regression test for Gambit input/output. """
    po_def = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]])
    po_att = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]])

    temp_dir = tempfile.TemporaryDirectory()

    # Convert pay-off matrices into a normal form game file.
    gambit_input_path = osp.join(temp_dir.name, "game.nfg")
    gambit_ops.encode_gambit_file(po_def, po_att, gambit_input_path)

    # Run gambit analysis on the normal form game resulting in a Nash equilibrium file.
    gambit_output_path = osp.join(temp_dir.name, "nash.txt")
    gambit_ops.gambit_analysis(100, gambit_input_path, gambit_output_path)

    # Decode Nash file.
    nash_att, nash_def = gambit_ops.decode_gambit_file(gambit_output_path)
    np.testing.assert_array_equal(nash_att, [0.5, 0.5, 0, 0])
    np.testing.assert_array_equal(nash_def, [0.5, 0.5, 0, 0])
    temp_dir.cleanup()


if __name__ == "__main__":
    pytest.main(["-s"])
