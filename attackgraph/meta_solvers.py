import numpy as np

class meta_method_selector(object):
    def __init__(self, meta_method_name):
        if meta_method_name == 'nash':
            self.meta_method = self.double_oracle
        elif meta_method_name == 'uniform':
            self.meta_method = self.uniform
        elif meta_method_name == 'SP':
            self.meta_method = self.self_play
        elif meta_method_name == 'DO_uniform':
            self.meta_method = self.DO_uniform
        elif meta_method_name == 'weighted_ne':
            self.meta_method = self.weighted_ne
        else:
            raise ValueError

    def sample(self, game, epoch):
        return self.meta_method(game, epoch)

    def double_oracle(self, game, epoch):
        mix_str_def = game.nasheq[epoch][0]
        mix_str_att = game.nasheq[epoch][1]
        return mix_str_def, mix_str_att

    def uniform(self, game, epoch):
        l = len(game.nasheq[epoch][0])
        mix_str_def = np.ones(l) / l
        mix_str_att = np.ones(l) / l
        return mix_str_def, mix_str_att

    def self_play(self, game, epoch):
        mix_str_def = np.zeros(len(game.nasheq[epoch][0]))
        mix_str_def[-1] = 1
        mix_str_att = np.zeros(len(game.nasheq[epoch][1]))
        mix_str_att[-1] = 1
        return mix_str_def, mix_str_att

    def DO_uniform(self, game, epoch):
        if epoch % 2 == 0:
            mix_str_def = game.nasheq[epoch][0]
            mix_str_att = game.nasheq[epoch][1]
        else:
            l = len(game.nasheq[epoch][0])
            mix_str_def = np.ones(l) / l
            mix_str_att = np.ones(l) / l
        return mix_str_def, mix_str_att


    def weighted_ne(self, game, epoch, gamma=0.4):
        # gamma is the discount factor for NEs.
        mix_str_def = np.zeros(epoch)
        mix_str_att = np.zeros(epoch)
        for i in np.arange(1, epoch+1):
            temp = game.nasheq[i][0].copy()
            mix_str_def[:len(temp)] += temp * gamma**(epoch-i)
            temp = game.nasheq[i][1].copy()
            mix_str_att[:len(temp)] += temp * gamma**(epoch-i)
        mix_str_def = mix_str_def / np.sum(mix_str_def)
        mix_str_att = mix_str_att / np.sum(mix_str_att)

        return mix_str_def, mix_str_att




# Payoff analysis
def payoff(game, nash_att, nash_def):
    num_str = len(nash_def)
    nash_def = np.reshape(nash_def, newshape=(num_str, 1))

    dPayoff = np.round(np.sum(nash_def * game.payoffmatrix_def * nash_att), decimals=2)
    aPayoff = np.round(np.sum(nash_def * game.payoffmatrix_att * nash_att), decimals=2)

    return dPayoff, aPayoff

def mean_payoff(game, nash_att, nash_def):
    num_str = len(nash_def)
    nash_def = np.reshape(nash_def, newshape=(num_str, 1))

    dPayoff = np.round(np.mean(np.sum(game.payoffmatrix_def * nash_att, axis=1)), decimals=2)
    aPayoff = np.round(np.mean(np.sum(nash_def * game.payoffmatrix_att, axis=0)), decimals=2)

    return dPayoff, aPayoff











