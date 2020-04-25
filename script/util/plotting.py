import numpy as np
import matplotlib.pyplot as plt


def plot_grid_map(data, title, cmap=plt.cm.Blues):
    fig, ax = plt.subplots(
        subplot_kw={'xticklabels': [], 'yticklabels': []})
    
    ax.set_title(title)
    ax.matshow(data, cmap=cmap)

    return fig