
import matplotlib.pyplot as plt


def std_plot(x, mean, std, ax=None, **kwargs):

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    ax.fill_between(x, mean - std, mean + std, **kwargs)

    return ax


def minmax_plot(x, min, max, ax=None, **kwargs):

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    ax.fill_between(x, min, max, **kwargs)

    return ax
