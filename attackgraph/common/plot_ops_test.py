""" Test suite for `plot_ops`. """
import os.path as osp

import dill as pickle
import pytest

from attackgraph import settings
from attackgraph.common import plot_ops


def test_payoff_matrix():
    """ Basic API test for `payoff_matrix`. """
    testdata_dir = osp.join(settings.SRC_DIR, "attackgraph", "testing")

    attacker = pickle.load(open(osp.join(testdata_dir, "payoff_attacker_yongzhao.pkl"), "rb"))
    defender = pickle.load(open(osp.join(testdata_dir, "payoff_defender_yongzhao.pkl"), "rb"))

    render = plot_ops.payoff_matrix(defender[42:52, 42:52], attacker[42:52, 42:52])
    print(render)


if __name__ == "__main__":
    pytest.main(["-s"])
