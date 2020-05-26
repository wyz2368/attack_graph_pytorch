""" Has action mask. """
import random

import numpy as np

from attackgraph import sample_strategy as ss


class Attacker(object):

    def __init__(self, G, oredges, andnodes, actionspace):
        self.num_nodes = G.number_of_nodes()
        self.observation = [0]*self.num_nodes
        self.canAttack = []
        self.attact = set()
        self.ORedges = oredges
        self.ANDnodes = andnodes
        self.actionspace = actionspace

    # When we're the opponent generate the action set.
    def att_greedy_action_builder(self, G, timeleft):
        self.attact.clear()
        isDup = False
        mask = np.array([self.get_att_canAttack_mask(G)], dtype=np.float32)

        observations = []
        actions = []

        while not isDup:
            att_input = self.att_obs_constructor(G, timeleft)
            observations += [att_input]
            x = self.nn_att(
                observation=att_input[None],
                stochastic=True,
                mask=mask,
                update_eps=-1,
                training_attacker=True)[0]
            actions += [x]
            action = self.actionspace[x]
            if action == 'pass':
                break
            isDup = (action in self.attact)
            if not isDup:
                self.attact.add(action)

    # Simulation.
    def att_greedy_action_builder_single(self, G, timeleft, nn_att):
        self.attact.clear()
        isDup = False
        mask = np.array([self.get_att_canAttack_mask(G)], dtype=np.float32)
        while not isDup:
            att_input = self.att_obs_constructor(G, timeleft)
            x = nn_att(
                observation=att_input[None],
                stochastic=False,
                mask=mask,
                update_eps=-1,
                training_attacker=True)[0]
            action = self.actionspace[x]
            if action == 'pass':
                break
            isDup = (action in self.attact)
            if not isDup:
                self.attact.add(action)

    # construct the input of the neural network
    def att_obs_constructor(self, G, timeleft):
        canAttack, inAttackSet = self.get_att_canAttack_inAttackSet(G)
        att_input = self.observation + canAttack + inAttackSet + [timeleft]
        return np.array(att_input, dtype=np.float32)

    # This function can also be used as masking illegal actions.
    def get_att_canAttack(self, G):
        canAttack = []
        # TODO: recheck the logics
        for andnode in self.ANDnodes:
            if G.nodes[andnode]['root'] == 1 and G.nodes[andnode]['state'] == 0:
                canAttack.append(1)
                continue
            if G.nodes[andnode]['root'] == 1 and G.nodes[andnode]['state'] == 1:
                canAttack.append(0)
                continue
            precondflag = 1
            precond = G.predecessors(andnode)
            for prenode in precond:
                if G.nodes[prenode]['state'] == 0:
                    precondflag = 0
                    break
            if G.nodes[andnode]['state'] == 0 and precondflag:
                canAttack.append(1)
            else:
                canAttack.append(0)

        for (father, son) in self.ORedges:
            if G.nodes[father]['state'] == 1 and G.nodes[son]['state'] == 0:
                canAttack.append(1)
            else:
                canAttack.append(0)

        return canAttack

    def get_att_canAttack_inAttackSet(self, G):
        # TODO: Check if attact includes illegal actions besides andnodes and edges.
        canAttack = []
        inAttackSet = []
        for andnode in self.ANDnodes:
            if andnode in self.attact:
                inAttackSet.append(1)
            else:
                inAttackSet.append(0)

            if G.nodes[andnode]['root'] == 1 and G.nodes[andnode]['state'] == 0:
                canAttack.append(1)
                continue
            if G.nodes[andnode]['root'] == 1 and G.nodes[andnode]['state'] == 1:
                canAttack.append(0)
                continue
            precondflag = 1
            precond = G.predecessors(andnode)
            for prenode in precond:
                if G.nodes[prenode]['state'] == 0:
                    precondflag = 0
                    break
            if G.nodes[andnode]['state'] == 0 and precondflag:
                canAttack.append(1)
            else:
                canAttack.append(0)

        for (father, son) in self.ORedges:
            if (father, son) in self.attact:
                inAttackSet.append(1)
            else:
                inAttackSet.append(0)
            if G.nodes[father]['state'] == 1 and G.nodes[son]['state'] == 0:
                canAttack.append(1)
            else:
                canAttack.append(0)

        return canAttack, inAttackSet

    def uniform_strategy(self, G, rand_limit):
        # TODO: rand_limit should be less than number of available actions.
        actmask = self.get_att_canAttack(G)
        attSet = list(self.ANDnodes) + self.ORedges
        actset_masked = list(x for x, z in zip(attSet, actmask) if z)
        return set(random.sample(actset_masked, min(rand_limit, len(actset_masked))))

    def update_obs(self, obs):
        self.observation = obs

    def reset_att(self):
        self.observation = [0]*self.num_nodes
        self.canAttack = []
        self.attact.clear()

    def update_canAttack(self, obs):
        self.canAttack = obs

    # Designed for mask function
    def get_att_canAttack_mask(self, G):
        canAttack = []
        # TODO: recheck the logics
        for andnode in self.ANDnodes:
            if G.nodes[andnode]['root'] == 1 and G.nodes[andnode]['state'] == 0:
                canAttack.append(0)
                continue
            if G.nodes[andnode]['root'] == 1 and G.nodes[andnode]['state'] == 1:
                canAttack.append(-1000)
                continue
            precondflag = 1
            precond = G.predecessors(andnode)
            for prenode in precond:
                if G.nodes[prenode]['state'] == 0:
                    precondflag = 0
                    break
            if G.nodes[andnode]['state'] == 0 and precondflag:
                canAttack.append(0)
            else:
                canAttack.append(-1000)

        for (father, son) in self.ORedges:
            if G.nodes[father]['state'] == 1 and G.nodes[son]['state'] == 0:
                canAttack.append(0)
            else:
                canAttack.append(-1000)

        canAttack.append(0)  # mask pass with 0.
        return canAttack

    def get_att_isActive(self, G):
        isActive = []
        for id in G.nodes:
            if G.nodes[id]['state'] == 1:
                isActive.append(1)
            else:
                isActive.append(0)
        return isActive

    def set_env_belong_to(self, env):
        self.myenv = env

    def set_mix_strategy(self, mix):
        self.mix_str = mix

    def set_str_set(self, set):
        self.str_set = set

    def sample_and_set_str(self, str_dict=None):
        assert self.myenv.training_flag == 0  # Should be training the attacker when sampling defender meta-strategy.
        nn, nn_name = ss.sample_strategy_from_mixed(env=self.myenv, str_set=self.str_set, mix_str=self.mix_str, identity=1, str_dict=str_dict)
        self.nn_name = nn_name
        self.nn_att = nn
