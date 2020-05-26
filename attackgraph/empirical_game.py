""" Emperical simulated game, EGTA-style. Environemnt is in DagGenerator """
import copy
import logging
import os.path as osp

import gin
import numpy as np

import attackgraph.common.file_ops as fp
from attackgraph import DagGenerator, settings

logger = logging.getLogger(__name__)


def init_game(saved_env_name: str = None, env_name: str = None):
    """ First attempts to load a saved environment, if not builds new enviornment.

    :param saved_env_name: Name of saved environment name to load.
    :param env_name: Name of environment to create.
    """
    assert (saved_env_name is not None) or (env_name is not None)

    if saved_env_name is not None:
        logger.info(f"Loading environment: {saved_env_name}")
        path = osp.join(settings.get_env_data_dir(), f"{saved_env_name}.pkl")
        if not osp.exists(path):
            raise ValueError("The env being loaded does not exist.")
        env = fp.load_pkl(path)

    else:
        env = DagGenerator.get_env_data_dir(env_name)

    # save graph copy
    env.save_graph_copy()
    env.save_mask_copy()  # TODO: change transfer

    # create players and point to their env
    env.create_players()
    env.create_action_space()

    # initialize game data
    game = EmpiricalGame(env)
    game.env.defender.set_env_belong_to(game.env)
    game.env.attacker.set_env_belong_to(game.env)

    # make no sense
    env.defender.set_env_belong_to(env)
    env.attacker.set_env_belong_to(env)

    return game


@gin.configurable
class EmpiricalGame(object):

    def __init__(self, env, num_episodes, threshold):
        # TODO: check if env should be initial env, this env should be with G_reserved.
        logger.info("Reminder: env in game should be same as the initial one since G should be G_reserved.")
        self.env = copy.deepcopy(env)
        self.att_str = []
        self.def_str = []
        self.nasheq = {}
        self.payoffmatrix_def = np.zeros((1, 1), dtype=np.float32)
        self.payoffmatrix_att = np.zeros((1, 1), dtype=np.float32)
        self.dir_def = settings.get_defender_strategy_dir()
        self.dir_att = settings.get_attacker_strategy_dir()

        # payoff and beneficial deviation
        self.att_BD_list = []
        self.def_BD_list = []
        self.att_payoff = []
        self.def_payoff = []

        self.num_episodes = num_episodes
        self.threshold = threshold

    def set_hado_param(self, param):
        self.param = param  # k, gamma, alpha = param

    def set_hado_time_step(self, steps):
        self.hado_time_step = steps

    def num_str(self):
        return len(self.att_str), len(self.def_str)

    def dim_payoff_def(self):
        return np.shape(self.payoffmatrix_def)

    def dim_payoff_att(self):
        return np.shape(self.payoffmatrix_att)

    def add_att_str(self, str_name):
        if not fp.isExist(osp.join(self.dir_att, str_name)):
            raise ValueError("This strategy does not exist.")
        if 'att_' not in str_name:
            raise ValueError("This may not be an attacker's strategy due to no att_ sign")
        if not isinstance(str_name, str):
            raise ValueError("The name to be added is not a str.")
        self.att_str.append(str_name)
        logger.info(str_name + " has been added to attacker's strategy set")

    def add_def_str(self, str_name):
        if not fp.isExist(osp.join(self.dir_def, str_name)):
            raise ValueError("This strategy does not exist.")
        if 'def_' not in str_name:
            raise ValueError("This may not be a defender's strategy due to no def_ sign")
        if not isinstance(str_name, str):
            raise ValueError("The name to be added is not a str.")
        self.def_str.append(str_name)
        logger.info(str_name + " has been added to defender's strategy set")

    def init_payoffmatrix(self, payoff_def, payoff_att):
        self.payoffmatrix_def[0, 0] = payoff_def
        self.payoffmatrix_att[0, 0] = payoff_att
        logger.info("Payoff matrix has been initilized by" + " " + str(payoff_def) + " for the defender.")
        logger.info("Payoff matrix has been initilized by" + " " + str(payoff_att) + " for the attacker.")

    def add_nasheq(self, ne_name, ne):  # ne is a dic。 nash is a numpy. 0: def, 1: att
        if not isinstance(ne_name, int):
            raise ValueError("The ne name to be added is not an integer.")
        if not isinstance(ne, dict):
            raise ValueError("The ne to be added is not a dictionary.")
        self.nasheq[ne_name] = ne

    def add_col_att(self, col):
        num_row, _ = np.shape(self.payoffmatrix_att)
        num_row_new, _ = np.shape(col)
        if num_row != num_row_new:
            raise ValueError("Cannot extend attacker column since dim does not match")
        self.payoffmatrix_att = np.append(self.payoffmatrix_att, col, 1)

    def add_row_att(self, row):
        _, num_col = np.shape(self.payoffmatrix_att)
        _, num_col_new = np.shape(row)
        if num_col != num_col_new:
            raise ValueError("Cannot extend attacker row since dim does not match")
        self.payoffmatrix_att = np.append(self.payoffmatrix_att, row, 0)

    def add_col_def(self, col):
        num_row, _ = np.shape(self.payoffmatrix_def)
        num_row_new, _ = np.shape(col)
        if num_row != num_row_new:
            raise ValueError("Cannot extend defender column since dim does not match")
        self.payoffmatrix_def = np.append(self.payoffmatrix_def, col, 1)

    def add_row_def(self, row):
        _, num_col = np.shape(self.payoffmatrix_def)
        _, num_col_new = np.shape(row)
        if num_col != num_col_new:
            raise ValueError("Cannot extend defender row since dim does not match")
        self.payoffmatrix_def = np.append(self.payoffmatrix_def, row, 0)

    def hado_str(self, identity, param):
        k, gamma, alpha = param
        num_ne = len(self.nasheq.keys())
        delta = self.nasheq[num_ne][identity].copy()
        denom = 0
        for i in np.arange(num_ne-1):
            temp = self.nasheq[i+1][identity].copy()
            temp.resize(delta.shape)
            delta += gamma**(num_ne-1-i)*temp
            denom += gamma**(num_ne-1-i)

        denom += 1
        return delta/denom

    # nash is a dictionary where nash[0] is defender's NE str and nash[1] is attacker's NE str.
    def regret(self):
        nash = self.nasheq[len(self.def_str)]
        nash_def = nash[0]
        nash_att = nash[1]
        num_str = len(nash_att)
        x1, y1 = np.shape(self.payoffmatrix_def)
        x2, y2 = np.shape(self.payoffmatrix_att)
        if x1 != y1 or x1 != x2 or x2 != y2 or x1 != num_str:
            raise ValueError("Dim of NE does not match payoff matrix.")

        nash_def = np.reshape(nash_def, newshape=(num_str, 1))

        dPayoff = np.round(np.sum(nash_def * self.payoffmatrix_def * nash_att), decimals=2)
        aPayoff = np.round(np.sum(nash_def * self.payoffmatrix_att * nash_att), decimals=2)

        utils_def = np.round(np.sum(self.payoffmatrix_def * nash_att, axis=1), decimals=2)
        utils_att = np.round(np.sum(nash_def * self.payoffmatrix_att, axis=0), decimals=2)

        regret_def = utils_def - dPayoff
        regret_att = utils_att - aPayoff

        regret_def = np.reshape(regret_def, newshape=np.shape(regret_att))

        regret_att = -regret_att
        regret_def = -regret_def

        return regret_att, regret_def

    def mean_regret(self):
        regret_att, regret_def = self.regret()
        mean_reg_att = np.round(np.mean(regret_att[1:]), decimals=2)
        mean_reg_def = np.round(np.mean(regret_def[1:]), decimals=2)
        return mean_reg_att, mean_reg_def
