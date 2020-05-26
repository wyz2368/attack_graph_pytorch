import os

import attackgraph.common.file_ops as fp

game_path = os.getcwd() + '/empirical_game/game.pkl'

game = fp.load_pkl(game_path)

print(game.att_str)
print(game.def_str)
print(game.nasheq)
print(game.payoffmatrix_def)
print(game.payoffmatrix_att)
