from attackgraph.common import json_ops as jp
import numpy as np

def create_graph():
    num_nodes = 10
    graph = {}
    graph['nodes'] = []
    graph['edges'] = []

    save_path = "./"
    save_name = '10n11e.json'

    AND_nodes = [1, 2, 3]
    goal_nodes = [9, 10]
    edges = [(1, 4), (1, 6), (2, 4), (2, 5), (3, 5), (4, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 10)]
    for node in range(1, num_nodes+1):
        node_info = {
                      "id": node,
                      "topoPosition": node,
                      "nodeType": "NONTARGET" if node not in goal_nodes else "TARGET",
                      "actType": "AND" if node in AND_nodes else "OR",
                      "state": "INACTIVE",
                      "aReward": 0.0 if node not in goal_nodes else np.random.uniform(10, 20),
                      "dPenalty": 0.0 if node not in goal_nodes else np.random.uniform(-20, -40),
                      "dCost": np.random.uniform(-5, -2),
                      "posActiveProb": np.random.uniform(0.6, 1),
                      "posInactiveProb": np.random.uniform(0, 0.3),
                      "aActivationCost": np.random.uniform(-5, -2),
                      "aActivationProb": np.random.uniform(0.7, 1)
                    }
        graph['nodes'].append(node_info)

    for i, edge in enumerate(edges):
        edge_info = {
                      "id": i,
                      "srcID": edge[0],
                      "desID": edge[1],
                      "edgeType": "NORMAL",
                      "aActivationCost": np.random.uniform(-5, -2),
                      "aActivationProb": np.random.uniform(0.7, 1)
                    }
        graph['edges'].append(edge_info)

    jp.save_json_data(path=save_path+save_name, json_obj=graph)


create_graph()

