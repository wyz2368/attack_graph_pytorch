import copy
import os
import random
import time

import numpy as np
import tensorflow as tf

import attackgraph.common.file_ops as fp
# from attackgraph.sample_strategy import rand_att_str_generator, rand_def_str_generator
from attackgraph import DagGenerator as dag
from attackgraph import empirical_game
from attackgraph import rand_strategies_payoff as rp
from attackgraph import training
from attackgraph.parallel_sim import parallel_sim
from attackgraph.sample_strategy import sample_strategy_from_mixed
from attackgraph.util import set_global_seed
from baselines import deepq
from baselines.common import models
from baselines.deepq import load_action

env = dag.Environment(numNodes=5, numEdges=4, numRoot=2, numGoals=1)

nodeset = [1,2,3,4,5]
edgeset = [(1,2),(2,3),(2,4),(5,2)]

attr = {}
attr['nodes'] = nodeset
attr['edges'] = edgeset
attr['Nroots'] = [1,0,0,0,1]
attr['Ntypes'] = [0,0,0,1,0]
attr['NeTypes'] = [1,1,0,0,1]
attr['Nstates'] = [0,0,0,0,0]
attr['NaRewards'] = [0,0,0,3,0]
attr['NdPenalties'] = [0,0,0,-3,0]
attr['NdCosts'] = [-1,-1,-1,-1,-1]
attr['NaCosts'] = [-1,-1,-1,-1,-1]
attr['NposActiveProbs'] = [0.6,0.6,0.6,0.6,0.6]
attr['NposInactiveProbs'] =[0.2,0.2,0.2,0.2,0.2]
attr['NactProbs'] = [0.8,0.8,0.8,0.8,0.8]

attr['Eeids'] = [1,2,3,4]
attr['Etypes'] = [0,0,0,0]
attr['actProb'] = [0,0.9,0.9,0]
attr['Ecosts'] = [0,-1,-1,0]


# nodeset = [1,2]
# edgeset = [(1,2)]

env.DagGenerator_wo_attrs(nodeset, edgeset)
env.specifiedDAG(attr)
env.save_graph_copy()

# env.visualize()

# print(env.G.nodes.data()[4])
# print(env.G.edges.data())
# set_global_seed(5)
env.create_players()
game = empirical_game.EmpiricalGame(env,4,256,[256,256],400,0.1)

def co(game):
    env1 = copy.deepcopy(game.env)
    env1.set_training_flag(6)
    game.env.set_training_flag(7)
    print(env1.training_flag)
    print(game.env.training_flag)
    print(env1.training_flag is game.env.training_flag)

co(game)

#test attacker
# print(env.attacker.ORedges)
# print(env.attacker.ANDnodes)
# print(env.attacker.actionspace)
# print(env.attacker.get_att_canAttack_inAttackSet(env.G))
# print(env.attacker.uniform_strategy(env.G,1))
# env.attacker.update_canAttack(env.attacker.get_att_canAttack(env.G))
# print(env.attacker.canAttack)
# env.attacker.reset_att()
# print(env.attacker.canAttack)

#test defender
# print(env.defender.num_nodes)
# print(env.defender.observation)
# print(env.defender.history)
# print(env.defender.prev_obs)
# print(env.defender.defact)
# print(env.defender.prev_defact)
# print(env.defender.rand_limit)

# env.defender.defact.add(2)
# env.defender.defact.add(3)
# env.defender.defact.add(5)
#
# print(env.defender.get_def_wasDefended(env.G))
# print(env.defender.get_def_inDefenseSet(env.G))
# print(env.defender.get_def_actionspace(env.G))
# print(env.defender.uniform_strategy(env.G))
#
# env.defender.update_obs([0,0,0,0,1])
# env.defender.update_obs([0,0,0,1,1])
# print(env.defender.observation)
#
# env.defender.save_defact2prev()
#
# print('*******')
# print(env.defender.observation)
# print(env.defender.prev_obs)
# print(env.defender.defact)
# print(env.defender.prev_defact)
#
# print(env.defender.def_obs_constructor(env.G,9))


#test the environment
# a = [(1,2),(5,9),(4,3),(1,9),(2,3)]
# print(env.sortEdge(a))
# print(env.getHorizon_G())
# print(env.G.nodes.data())
# print(env.isOrType_N(5))
# print(env.G.nodes)
# for i in env.G.nodes:
    # print(env.getState_N(i))
    # print(env.getType_N(i))
    # print(env.getActivationType_N(i))
    # print(env.getAReward_N(i))
    # print(env.getDPenalty_N(i))
    # print(env.getDCost_N(i))
    # print(env.getACost_N(i))
    # print(env.getActProb_N(i))
    # print(env.getposActiveProb_N(i))
    # print(env.getposInactiveProb_N(i))

# print(env.G.edges)
# for i in env.G.edges:
#     # print(env.getid_E(i))
#     print(env.getActProb_E(i))

# env.print_N(1)
# env.print_E((2,3))

# print(env.getNumNodes())
# print(env.getNumEdges())
# for i in env.G.nodes:
    # print(env.inDegree(i))
    # print(env.outDegree(i))
    # print(env.predecessors(i))
    # print(env.successors(i))

# print(env.isDAG())
# print(env.getEdges())
# print(env.get_ANDnodes())
# print(env.get_ORnodes())
# print(env.get_ORedges())
# print(env.get_Targets())
# print(env.get_Roots())
# print(env.get_NormalEdges())
# print(env.get_att_isActive())
# print(env.get_def_hadAlert())
# print(env.get_att_actionspace())
# print(env.get_def_actionspace())

# a = [1,2,3]
# print(env.check_nodes_sorted(a))


# test mask
# def mask_generator_att(env, obses):
#     batch_size = np.shape(obses)[0]
#     num_nodes = env.G.number_of_nodes()
#     mask = []
#     for i in np.arange(batch_size):
#         state = obses[i][:num_nodes]
#         G_cur = env.G_reserved.copy()
#
#         for j in G_cur.nodes:
#             G_cur.nodes[j]['state'] = state[j-1]
#
#         _mask = env.attacker.get_att_canAttack_mask(G_cur)
#
#         mask.append(_mask)
#     return np.array(mask)
#
# obses = np.array([[1,0,0,0,0],[0,0,0,0,1],[1,0,0,0,1]])
#
# mask = mask_generator_att(env, obses)
# print(mask)

# Test sim using random strategies

# t1 = time.time()
# # payoff_att, payoff_def, ta, tb, tc = rp.parallel_sim(env,1000)
# a,b  = rp.rand_parallel_sim(env,1000)
# t2 = time.time()
#
# t3 = time.time()
# payoff_att, payoff_def, tz, tx = rp.rand_strategies_payoff(env,1000)
# t4 = time.time()
#
# # print(payoff_def,payoff_att)
#
# # print(t2-t1,t4-t3, ta, tb, tc)
#
# # print(tz,tx)
#
# print(t2-t1,t4-t3)
# # print(a,b)

#Test creating new random strategies

# rand_att_str_generator(env,game)
# rand_def_str_generator(env,game)

# Test load action
# path = os.getcwd() + "/attacker_policies/att_str_epoch1.pkl"
# training_flag = 1
# act = load_action.load_action(path,game,training_flag)
# print(type(act))

# Test sample mixed strategy
# str_set = ['1.pkl', '2.pkl', '3.pkl']
# mix_str = np.array([0.3,0.3,0.4])
# identity = 0
# sample_strategy_from_mixed(env, str_set, mix_str, identity)


# Test sim using two networks.
# path = os.getcwd() + "/attacker_policies/att_str_epoch1.pkl"
# training_flag = 1
# act_att = load_action.load_action(path,game,training_flag)
#
# env.attacker.att_greedy_action_builder_single(env.G,timeleft=8,nn_att=act_att)
# print(env.attacker.attact)
#
# path = os.getcwd() + "/defender_policies/def_str_epoch1.pkl"
# training_flag = 0
# act_def = load_action.load_action(path,game,training_flag)
# env.defender.def_greedy_action_builder_single(env.G,timeleft=8,nn_def=act_def)
# print(env.defender.defact)

# act_att = 'att_str_epoch1.pkl'
# act_def = 'def_str_epoch1.pkl'
#
# out = parallel_sim(env, game, nn_att=act_att, nn_def=act_def, num_episodes=2)
# print(out)

# num_actions_def = env.act_dim_def()
# num_actions_att = env.act_dim_att()
# print(num_actions_def)
# print(num_actions_att)
# obs = env.attacker.att_obs_constructor(env.G, 8)
# print(len(obs))

# Test Training
