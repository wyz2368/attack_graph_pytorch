""" Lazy loading neural network.

This module lazily-loads and caches a PyTorch module from a saved file.
We need this primarily because we keep track of policies via file paths.
"""
import logging
import os.path as osp
from dataclasses import dataclass

import gin
import torch

logger = logging.getLogger(__name__)


@gin.configurable
@dataclass
class LazyLoadNN(object):
    f"""{__doc__}"""

    save_path: str

    def __post_init__(self):
        self.net = None
        assert osp.exists(self.save_path), f"Cannot find file: {self.save_path}"

    def _lazy_load(self):
        """ Lazily load the neural network's state. """
        # If the network is already loaded, do nothing.
        if self.net is not None:
            return

        # Otherwise, we need to load the network.
        else:
            logger.debug(f"Loading policy at: {self.save_path}")
            self.net = torch.load(self.save_path)

    def __call__(self, *args, **kwargs):
        """ Call the underlying neural network.

        :return: The output of a forward call to the neural network.
        """
        self._lazy_load()
        return self.net(*args, **kwargs)
