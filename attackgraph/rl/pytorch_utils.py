""" PyTorch utility functions. """
import numpy as np
import torch


def identity(x):
    return x


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def send_to_gpu(module, device: int = None):
    if isinstance(module, np.ndarray):
        module = torch.Tensor(module)

    # Send to no GPU, and leave on CPU.
    if device is None:
        return module

    # If we attempted to send it to GPU that doesn't exist.
    if not torch.cuda.is_available():
        raise ValueError("Tried to use GPU with no devices available.")

    # Send to GPU device.
    return module.to(torch.device(f"cuda:{device}"))
