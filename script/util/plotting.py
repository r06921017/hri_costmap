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

def plot_policy(grid, policy, title, values=None, cmap=plt.cm.Blues):
    arrows = [
        u'→',
        u'←',
        u'↓',
        u'↑',
    ]

    if values is None:
        values = np.zeros_like(grid)

    fig, ax = plt.subplots(
        subplot_kw={'xticklabels': [], 'yticklabels': []})

    ax.matshow(values, cmap=cmap)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            act = policy.get_action(np.ravel_multi_index([j, i], grid.shape))
            arrow = arrows[act]
            ax.text(j, i, arrow, ha='center', va='center', color='r')
    ax.set_title(title)

    return fig

def plot_q_function(Q, title):
    pass
