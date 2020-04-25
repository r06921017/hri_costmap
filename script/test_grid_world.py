import time

import numpy as np
import matplotlib.pyplot as plt
from mdptoolbox.mdp import ValueIteration

from grid_world import GridWorld
from util.plotting import plot_grid_map


def test_gridworld_value_iteration():
    N = 10
    grid = np.zeros((N, N), dtype=int)
    grid[:N-1, N-1] = 1  # Add obstacles
    env = GridWorld(
        init_pos=(0, 0),
        goal_pos=(N-1, N-1),
        human_pos=(4, 4),
        render=True,
        action_success_rate=1,
        grid=grid)
    
    gamma = 0.9
    vi = ValueIteration(env.T, env.R, gamma)
    vi.run()
    
    pi = vi.policy
    obs, rew, done, info = env.reset()
    while not done:
        act = pi[obs]
        obs, rew, done, info = env.step(act)
        time.sleep(0.5)

    R = env.R.reshape((N, N)).T
    V = np.asarray(vi.V).reshape((N, N)).T

    # plot results
    fig1 = plot_grid_map(R, "Reward (Ground Truth)", cmap=plt.cm.Reds)
    fig2 = plot_grid_map(V, "Value Function", cmap=plt.cm.Blues)
    plt.show()


if __name__ == '__main__':
    test_gridworld_value_iteration()
