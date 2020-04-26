# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plot_grid_map(data, title, cmap=plt.cm.Blues):
    fig, ax = plt.subplots(
        subplot_kw={'xticklabels': [], 'yticklabels': []})
    
    ax.matshow(data, cmap=cmap)
    ax.set_title(title)

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
            anno = ax.text(j, i, arrow, ha='center', va='center', color='r')
    ax.set_title(title)

    return fig

def plot_q_function(Q, title):
    pass
