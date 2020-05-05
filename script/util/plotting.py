# -*- coding: utf-8 -*-
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import make_axes_locatable


def plot_grid_map(data, title, print_values=False, cmap='bone'):
    fig, ax = plt.subplots(
        subplot_kw={'xticklabels': [], 'yticklabels': []})
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='5%', pad=0.1)
    
    im = ax.matshow(data, cmap=cmap)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(title)

    if print_values:
        for i, j in product(range(data.shape[0]), range(data.shape[1])):
            value = "{:2.2f}".format(data[i, j])
            ax.text(j, i, value, ha='center', va='center', color='r')

    return fig

def plot_policy(policy, dims, title, values=None, cmap='bone'):
    arrows = [
        u'→',
        u'←',
        u'↓',
        u'↑',
    ]

    if values is None:
        values = np.zeros(dims)

    fig, ax = plt.subplots(
        subplot_kw={'xticklabels': [], 'yticklabels': []})

    ax.matshow(values, cmap=cmap)
    for i in range(dims[0]):
        for j in range(dims[1]):
            act = policy.get_action(np.ravel_multi_index([j, i], dims))
            arrow = arrows[act]
            ax.text(j, i, arrow, ha='center', va='center', color='r')
    ax.set_title(title)

    return fig

def plot_q_function():
    # TODO: use plt.triplot()
    pass

def plot_dataset_distribution(dataset, dims, title, normalize=False, cmap='bone'):
    dataset_density = np.zeros(dims[0] * dims[1])

    for t in dataset:
        for trans in t:
            dataset_density[trans.obs] += 1
    
    if normalize:
        dataset_density /= np.sum(dataset_density)
    dataset_density = dataset_density.reshape(dims).T

    fig = plot_grid_map(dataset_density, title, cmap=cmap)
    return fig
