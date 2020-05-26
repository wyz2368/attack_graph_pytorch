""" Functions for constructing neural network policies.

Resources:
 - https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/networks.py
"""
import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

import attackgraph.rl.pytorch_utils as ptu
from attackgraph.rl.modules.layer_norm import LayerNorm


@gin.configurable
class MLP(nn.Module):
    """ Multi-Layer Perceptron. """

    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size,
            init_w: float = 3e-3,
            hidden_activation=F.relu,
            output_activation=ptu.identity,
            hidden_init=ptu.fanin_init,
            b_init_value: float = 0.1,
            layer_norm: bool = False,
            layer_norm_kwargs: dict = {}):
        nn.Module.__init__(self)

        self.layers = []
        self.layer_norms = []
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        prev_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            layer = nn.Linear(prev_size, next_size)
            # Set the initial weights of the layer.
            hidden_init(layer.weight)
            layer.bias.data.fill_(b_init_value)

            self.__setattr__("layer_{}".format(i), layer)
            self.layers.append(layer)

            # Optionally apply layer normalization.
            if layer_norm:
                layer_norm = LayerNorm(prev_size)
                self.__setattr__("layer_norm_{}".format(i), layer_norm)
                self.layer_norms.append(layer_norm)

            # Prepare for next layer.
            prev_size = next_size

        # Create output layer.
        self.last_layer = nn.Linear(prev_size, output_size)
        self.last_layer.weight.data.uniform_(-init_w, init_w)
        self.last_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)

            if self.layer_norm and i < len(self.layers) - 1:
                h = self.layer_norms[i](h)

            h = self.hidden_activation(h)

        h = self.last_layer(h)
        y = self.output_activation(h)
        return y


class FlattenMLP(MLP):
    """ Flatten inputs along dim 1 and pass it through a MLP. """

    def forward(self, *inputs):
        flat_x = torch.cat(inputs, dim=1)
        return super().forward(flat_x)
