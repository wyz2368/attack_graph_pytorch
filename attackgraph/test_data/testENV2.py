import os
import random

import numpy as np

import attackgraph.common.file_ops as fp
from attackgraph import DagGenerator as dag

# random.seed(2)
# np.random.seed(2)
# env = dag.Environment(numNodes=5, numEdges=4, numRoot=2, numGoals=1)
# env.randomDAG()
#
# print(env.G.nodes.data())
# print(env.G.edges)

# env = dag.env_rand_gen_and_save("test_env")
# print(env.G.nodes)
# print(env.G.edges)


def env_rand_gen(env_name, num_attr_N = 11, num_attr_E = 4, T=10, graphid=1, numNodes=30, numEdges=100, numRoot=5, numGoals=6, history = 3):
    env = dag.Environment(num_attr_N = num_attr_N, num_attr_E = num_attr_E, T=T, graphid=graphid, numNodes=numNodes, numEdges=numEdges, numRoot=numRoot, numGoals=numGoals, history = history)
    # env.randomDAG()
    env.load_graph()
    path = os.getcwd() + "/env_data/" + env_name + ".pkl"
    print("env path is ", path)
    fp.save_pkl(env,path)
    print(env_name + " has been saved.")
    return env

env = env_rand_gen('run_env_14isAND')
print(env.get_Roots())

# # _, targets = env.get_Targets()
# # print(targets)
# # for i in targets:
# #     print(i,':', list(env.G.successors(i)))
# # env.visualize()
# env = fp.load_pkl(os.getcwd() + "/env_data/" + 'run_env' + ".pkl")
# for i in env.G.nodes.data():
#     print(i)
#
# print('++++++++++++++++++++++++++++++++++')
#
# for i in env.G.edges.data():
#     print(i)
