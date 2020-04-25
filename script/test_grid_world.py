import time

import numpy as np
import matplotlib.pyplot as plt

from grid_world import GridWorld
import util.mdp as mdp
from util.plotting import plot_grid_map
from util.policy import EpsGreedyPolicy, BoltzmannPolicy


def test_gridworld_value_iteration():
    N = 10
    grid = np.zeros((N, N), dtype=int)
    grid[:N-1, N-1] = 1  # Add obstacles
    env = GridWorld(
        init_pos=(0, 0),
        goal_pos=(N-1, N-1),
        human_pos=(5, 5),
        render=True,
        action_success_rate=1,
        grid=grid)
    
    vi = mdp.value_iteration(env.T, env.R, gamma=0.99)
    policy = EpsGreedyPolicy(env.action_space(), vi)
    import pdb; pdb.set_trace()
    # TODO: replace with policy viz
    obs, rew, done, info = env.reset()
    while not done:
        act = policy.get_action(obs)
        obs, rew, done, info = env.step(act)
        time.sleep(0.2)

    R = env.R.reshape((N, N)).T
    V = np.asarray(vi.V).reshape((N, N)).T

    # plot results
    fig1 = plot_grid_map(R, "Reward (Ground Truth)", cmap=plt.cm.Reds)
    fig2 = plot_grid_map(V, "Value Function", cmap=plt.cm.Blues)
    plt.show()

def test_gridworld_q_learning():
    N = 5
    grid = np.zeros((N, N), dtype=int)
    grid[:N-1, N-1] = 1  # Add obstacles
    env = GridWorld(
        init_pos=(0, 0),
        goal_pos=(N-1, N-1),
        human_pos=(4, 0),
        render=True,
        action_success_rate=1,
        grid=grid)

    ql = mdp.q_learning(env.T, env.R, gamma=0.99)
    ql.run()

    policy = BoltzmannPolicy(env.action_space(), ql, tau=0.3)
    import pdb; pdb.set_trace()
    # TODO: replace with policy viz
    obs, rew, done, info = env.reset()
    while not done:
        act = policy.get_action(obs)
        obs, rew, done, info = env.step(act)
        time.sleep(0.2)


if __name__ == '__main__':
    test_gridworld_value_iteration()
    test_gridworld_q_learning()
