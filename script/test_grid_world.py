import time

import numpy as np
import matplotlib.pyplot as plt

from grid_world import GridWorld
from util.mdp import q_learning, value_iteration
from util.plotting import plot_grid_map, plot_policy
from util.policy import EpsGreedyPolicy, StochasticGreedyPolicy
from util.reward import construct_goal_reward, construct_human_radius_reward


def test_gridworld_value_iteration():
    np.random.seed(0)

    N = 10
    goal_pos = np.array([[N-1, N-1], [N-1, N-2]])
    human_pos = np.array([[N//2, N//2], [N-1, 0]])
    human_radius = 3

    grid = np.zeros((N, N), dtype=float)
    grid = construct_goal_reward(grid, goal_pos, 10)
    grid = construct_human_radius_reward(grid, human_pos, human_radius, -10)

    env = GridWorld(
        dimensions=(N, N),
        init_pos=(0, 0),
        goal_pos=goal_pos,
        reward_grid=grid,
        human_pos=human_pos,
        action_success_rate=1,
        render=True,
    )
    
    mdp_algo = value_iteration(env.transition, env.reward, gamma=0.99)
    policy = EpsGreedyPolicy(env.action_space(), mdp_algo)

    # plot results
    R = env.reward.reshape((N, N)).T
    V = np.asarray(mdp_algo.V).reshape((N, N)).T

    plot_grid_map(R, "Reward", cmap=plt.cm.Reds)
    plot_grid_map(V, "Value Function", cmap=plt.cm.Blues)
    plot_policy(policy, (N, N), "Policy", values=V, cmap=plt.cm.Blues)
    plt.show()

    obs, rew, done, info = env.reset()
    while not done:
        act = policy.get_action(obs)
        obs, rew, done, info = env.step(act)
        time.sleep(0.2)

    env.close()

def test_gridworld_q_learning():
    np.random.seed(0)

    N = 5
    goal_pos = np.array([[N-1, N-1]])
    human_pos = np.array([[N-1, 0]])
    human_radius = 2

    grid = np.ones((N, N), dtype=float) * -1
    grid = construct_goal_reward(grid, goal_pos, 10)
    grid = construct_human_radius_reward(grid, human_pos, human_radius, -10)

    env = GridWorld(
        dimensions=(N, N),
        init_pos=(0, 0),
        goal_pos=goal_pos,
        reward_grid=grid,
        human_pos=human_pos,
        action_success_rate=0.8,
        render=True,
    )

    mdp_algo = q_learning(env.transition, env.reward, gamma=0.99)
    mdp_algo.run()
    policy = StochasticGreedyPolicy(
        env.action_space(), mdp_algo, env.transition)

    # plot results
    R = env.reward.reshape((N, N)).T
    V = np.asarray(mdp_algo.V).reshape((N, N)).T

    plot_grid_map(R, "Reward", cmap=plt.cm.Reds)
    plot_grid_map(V, "Value Function", cmap=plt.cm.Blues)
    plt.show()

    obs, rew, done, info = env.reset()
    while not done:
        act = policy.get_action(obs)
        obs, rew, done, info = env.step(act)
        time.sleep(0.2)

    env.close()


if __name__ == '__main__':
    test_gridworld_value_iteration()
    test_gridworld_q_learning()
