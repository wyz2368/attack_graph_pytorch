""" Utility function to perform supervised learning in PyTorch. """
import logging
import typing
from dataclasses import dataclass

import gin
import numpy as np
import torch
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from attackgraph.policy_distillation import simulate_rewards
from attackgraph.rl.modules.lazy_load_nn import LazyLoadNN
from attackgraph.rl.modules.q_mixture import QMixtureWithOpponentClassifier


logger = logging.getLogger("attackgraph")


@dataclass
class MapDataset(torch.utils.data.Dataset):
    """ Maintains a dataset which is represented by a map/dict.

    Resources:
     - https://discuss.pytorch.org/t/dictionary-in-dataloader/40448
    """

    features: typing.Dict

    def __getitem__(self, index):
        datum = {}
        for key, value in self.features.items():
            datum[key] = value[index]
        return datum

    def __len__(self):
        key = list(self.features.keys())[0]
        return len(self.features[key])


@gin.configurable
def supervised_learning(net, train_X, train_Y, test_X, test_Y, criterion, n_epochs: int, eval_freq: int, batch_size: int, log_dir: str):
    """ Train a classifier using supervised learning. """

    train_dataset = MapDataset({"x": train_X, "y": train_Y})
    test_dataset = MapDataset({"x": test_X, "y": test_Y})

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Found {len(train_dataset)} training examples, making {len(train_loader)} batches.")
    logger.info(f"Found {len(test_dataset)} testing examples, making {len(test_loader)} batches.")

    optimizer = torch.optim.Adam(net.parameters())

    writer = SummaryWriter(logdir=log_dir)

    for epoch_i in range(n_epochs):
        logger.info(f"Epoch {epoch_i}")
        # Train ----------------------------------------------------------------
        net.train()
        for batch_i, batch in tqdm(enumerate(train_loader)):
            batch_i = epoch_i*len(train_loader) + batch_i

            prediction = net(batch["x"].float())

            optimizer.zero_grad()
            loss = criterion(prediction, batch["y"].long())
            writer.add_scalar("train/loss", loss.item(), batch_i)
            loss.backward()
            optimizer.step()

            accuracy_mean, accuracy_std = _accuracy(batch["y"].numpy(), prediction.detach().numpy())
            writer.add_scalar("train/accuracy/mean", accuracy_mean, batch_i)
            writer.add_scalar("train/accuracy/std", accuracy_std, batch_i)

            # Evaluation -----------------------------------------------------------
            if batch_i % eval_freq == 0:
                net.eval()
                total_eval_loss = 0.0
                n_eval_batches = 0.0

                for batch in test_loader:
                    prediction = net(batch["x"].float())
                    loss = criterion(prediction, batch["y"].long())
                    total_eval_loss += loss.item()
                    n_eval_batches += 1

                writer.add_scalar("eval/loss", total_eval_loss/n_eval_batches, batch_i)

                accuracy_mean, accuracy_std = _accuracy(batch["y"].numpy(), prediction.detach().numpy())
                writer.add_scalar("eval/accuracy/mean", accuracy_mean, batch_i)
                writer.add_scalar("eval/accuracy/std", accuracy_std, batch_i)

                # Get the reward against the mixture.
                reward_mean, reward_std = make_qmix_and_simulate(net)
                writer.add_scalar("eval/reward/mean", reward_mean, batch_i)
                writer.add_scalar("eval/reward/std", reward_std, batch_i)

                # Get rewards against pure-strategies.
                for opp_i in range(3):
                    with gin.config_scope(f"opp_{opp_i}"):
                        reward_mean, reward_std = make_qmix_and_simulate(net)
                        writer.add_scalar(f"eval/opp_{opp_i}/reward/mean", reward_mean, batch_i)
                        writer.add_scalar(f"eval/opp_{opp_i}/reward/std", reward_std, batch_i)

    return net


@gin.configurable
def make_qmix_and_simulate(classifier, q_funcs, opponent_paths, mixture, opponent_mixture=None):
    if opponent_mixture is None:
        opponent_mixture = mixture
    policy = QMixtureWithOpponentClassifier(
        mixture=mixture,
        q_funcs=[LazyLoadNN(x) for x in q_funcs],
        classifier=classifier)
    return simulate_rewards(policy, opponent_paths, opponent_mixture)


def _accuracy(target, predictions):
    predictions = np.argmax(predictions, axis=1)
    is_correct = predictions == target
    return np.mean(is_correct), np.std(is_correct)
