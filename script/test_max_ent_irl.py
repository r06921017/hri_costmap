from itertools import product
import time

import matplotlib.pyplot as plt
import numpy as np

from grid_world import GridWorld
from max_ent_irl import MaxEntIRL
from util.dataset import collect_trajectories
from util.mdp import q_learning, value_iteration
from util.plotting import plot_grid_map, plot_policy, plot_dataset_distribution
from util.policy import EpsGreedyPolicy, GreedyPolicy, StochasticGreedyPolicy
from util.reward import (construct_feature_boundary_reward,
    construct_goal_reward, construct_human_radius_reward)


def test_gridworld_maxent_irl():
    np.random.seed(0)

    # env
    N = 10
    goal_pos = np.array([[N-1, N-1]])
    human_pos = np.array([[3, 3]])
    human_radius = 2

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
        render=False,
    )

    # learn a policy
    mdp_algo = value_iteration(env.transition, env.reward, gamma=0.99)
    # mdp_algo = q_learning(env.transition, env.reward, gamma=0.99)
    # policy = GreedyPolicy(env.action_space(), mdp_algo)
    # policy = EpsGreedyPolicy(env.action_space(), mdp_algo, epsilon=0.1)
    policy = StochasticGreedyPolicy(
        env.action_space(), mdp_algo, env.transition)

    V = np.asarray(mdp_algo.V).reshape((N, N)).T
    R = env.reward.reshape((N, N)).T
    plot_grid_map(R, "Reward (Ground Truth)", cmap=plt.cm.Reds)
    plot_grid_map(V, "Value Function", cmap=plt.cm.Blues)

    # roll out trajectories
    dataset = collect_trajectories(
        policy=policy,
        env=env,
        num_trajectories=200,
        maxlen=N*2)
    plot_dataset_distribution(dataset, (N, N), "Dataset State Distribution")

    # feature map
    feature_map = [env._feature_map(s)
                   for s in range(env.observation_space().n)]
    feature_map = np.array(feature_map)

    # IRL
    me_irl = MaxEntIRL(
        observation_space=env.observation_space(),
        action_space=env.action_space(),
        transition=env.transition,
        goal_states=env.goal_states,
        dataset=dataset,
        feature_map=feature_map,
        max_iter=10,
        lr=0.1,
        anneal_rate=0.9)
    Rprime = me_irl.train()
    Rprime = Rprime.reshape((N, N)).T

    # plot results
    plot_grid_map(Rprime, "Reward (IRL)", print_values=True, cmap=plt.cm.Blues)
    plt.show()

def test_feature_gridworld_maxent_irl():
    np.random.seed(0)

    # env
    N = 15

    init_pos = np.zeros((N, N), dtype=float)
    for i, j in product(range(N//2+2, N), range(N//2+2, N)):
        init_pos[i, j] = i**2 + j**2
    init_pos /= np.sum(init_pos)
    goal_pos = np.array([[n, 0] for n in range(N)])

    grid = np.zeros((N, N), dtype=float)
    grid = construct_goal_reward(grid, goal_pos, 10)
    grid = construct_feature_boundary_reward(
        grid,
        boundary_axis=0,
        boundary_value=N//2,
        reward=-10,
        exp_constant=0.2,
    )

    plot_grid_map(init_pos.T, "Initial Position Distribution", cmap=plt.cm.Blues)
    plot_grid_map(grid.T, "Reward (Ground Truth)", cmap=plt.cm.Reds)

    env = GridWorld(
        dimensions=(N, N),
        init_pos=init_pos,
        goal_pos=goal_pos,
        reward_grid=grid,
        action_success_rate=1,
        render=False,
    )

    # learn a policy
    mdp_algo = value_iteration(env.transition, env.reward, gamma=0.99)
    policy = StochasticGreedyPolicy(
        env.action_space(), mdp_algo, env.transition)

    # roll out trajectories
    dataset = collect_trajectories(
        policy=policy,
        env=env,
        num_trajectories=20,
        maxlen=N*2)
    plot_dataset_distribution(dataset, (N, N), "Dataset State Distribution")

    # IRL feature map
    feature_map = [env._feature_map(s)
                   for s in range(env.observation_space().n)]
    feature_map = np.array(feature_map)

    # IRL
    me_irl = MaxEntIRL(
        observation_space=env.observation_space(),
        action_space=env.action_space(),
        transition=env.transition,
        goal_states=env.goal_states,
        dataset=dataset,
        feature_map=feature_map,
        max_iter=10,
        lr=0.1,
        anneal_rate=0.9)
    Rprime = me_irl.train()
    Rprime = Rprime.reshape((N, N)).T

    # plot results
    plot_grid_map(Rprime, "Reward (IRL)", cmap=plt.cm.Blues)
    plt.show()


if __name__ == '__main__':
    test_gridworld_maxent_irl()
    test_feature_gridworld_maxent_irl()
