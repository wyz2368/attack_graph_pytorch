""" Utility functions for plotting. """
import os.path as osp

import dill as pickle
import matplotlib.pylab as plt
import numpy as np


def generate_and_save_line_plot(path: str, title: str, x: str, y: str):
    """ Generates a line plot from a file.

    :param path: Path to a pickled list of data.
    :param tile: Plot title.
    :param x: Independent variable name.
    :param y: Dependent variable name.
    """
    fig, ax = plt.subplots()

    with open(path, "rb") as input_file:
        data = pickle.load(input_file)
        plt.plot(np.arange(len(data)), data)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    save_path = osp.join(osp.dirname(path), osp.basename(path))
    save_path = save_path[:-3] + "png"
    with open(save_path, "wb") as output_file:
        fig.savefig(output_file)


def generate_and_save_histogram(path: str, title: str, x: str, n_bins: int = 50):
    """ Generates a histogram from a file.

    :param path: Path to a pickled list of data.
    :param tile: Plot title.
    :param x: Independent variable name.
    :param n_bins: Number of bins.
    """
    fig, ax = plt.subplots()

    with open(path, "rb") as input_file:
        data = pickle.load(input_file)
        plt.hist(data, n_bins=n_bins)
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel("Frequency")

    save_path = osp.join(osp.dirname(path), osp.basename(path))
    save_path = save_path[:-3] + "png"
    with open(save_path, "wb") as output_file:
        fig.savefig(output_file)


def payoff_matrix(defender: np.ndarray, attacker: np.ndarray) -> str:
    """ Markdown render of a payoff matrix.

    :param defender:
    :param attacker:
    :return: String containing markdown rendering of payoff matrix.
    """
    assert defender.shape == attacker.shape
    n_rows, n_cols = defender.shape

    # Header of the table. Leaves the first column untitled, and labels the
    # remaining columns with attacker strategy indices.
    render = "||" + "".join([f"{x}|" for x in range(n_cols)]) + "\n"
    # Set column settings, default.
    render += "|" + "---|"*(n_cols+1) + "\n"

    # Body of the table.
    for row in range(n_rows):
        # Label row with defender's strategy.
        render += f"|{row}|"

        # Populate payoff cells.
        for col in range(n_cols):
            u_att = attacker[row][col]
            u_def = defender[row][col]
            render += f"{u_def:.1f}, {u_att:.1f}|"

        render += "\n"

    return render
