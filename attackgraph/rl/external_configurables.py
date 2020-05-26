""" Import external configurables for gin. """
import gin
import torch
import torch.nn.functional as F

gin.external_configurable(F.relu, "relu")
gin.external_configurable(F.tanh, "tanh")
gin.external_configurable(torch.optim.Adam, "torch.optim.Adam")
gin.external_configurable(torch.nn.CrossEntropyLoss, "torch.nn.CrossEntropyLoss")

@gin.configurable
def softmax(x):
    return F.softmax(x, dim=-1)
