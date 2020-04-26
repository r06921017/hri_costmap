import time

import matplotlib.pyplot as plt
import numpy as np

from dataset import Dataset, transition
from grid_world import GridWorld
from max_ent_irl import MaxEntIRL
from util.mdp import q_learning, value_iteration
from util.plotting import plot_grid_map, plot_policy
from util.policy import *


def test_gridworld_maxent_irl():
    np.random.seed(0)

    # env
    N = 5
    grid = np.zeros((N, N), dtype=int)
    grid[:N-1, N-1] = 1  # Add obstacles
    env = GridWorld(
        init_pos=(0, 0),
        goal_pos=(N-1, N-1),
        human_pos=(N-1, 0),
        human_radius=1.5,
        grid=grid,
        action_success_rate=1,
        render=False,
    )

    # learn a policy
    # ql = q_learning(env.T, env.R, gamma=0.99)
    # policy = BoltzmannPolicy(env.action_space(), ql, tau=1)
    vi = value_iteration(env.T, env.R, gamma=0.99)
    # policy = EpsGreedyPolicy(env.action_space(), vi, epsilon=0.1)
    # policy = StochasticGreedyPolicy(env.action_space(), vi, env.T)
    policy = GreedyPolicy(env.action_space(), vi)
    plot_policy(grid, policy, "Policy", values=np.asarray(vi.V).reshape((5,5)).T, cmap=plt.cm.Blues)
    plt.show()

    num_trajectories = 20
    maxlen = 15
    dataset = Dataset(maxlen=maxlen)

    # roll out trajectories
    for n in range(num_trajectories):
        t = []
        obs, rew, done, info = env.reset()
        policy.reset()

        for i in range(maxlen):
            act = policy.get_action(obs)
            next_obs, rew, done, info = env.step(act)
            # time.sleep(0.2)

            t.append(transition(obs=obs, act=act, next_obs=next_obs, rew=rew))

            if done:
                break
            obs = next_obs

        dataset.append(t)

    # phi
    phi = np.eye(env.observation_space().n, dtype=np.float)

    # IRL
    me_irl = MaxEntIRL(
        observation_space=env.observation_space(),
        action_space=env.action_space(),
        transition=env.T,
        goal_states=[env.goal_state],
        dataset=dataset,
        feature_map=phi,
        max_iter=5)
    Rprime = me_irl.train()
    Rprime = Rprime.reshape((N, N)).T

    # plot results
    R = env.R.reshape((N, N)).T

    plot_grid_map(R, "Reward (Ground Truth)", cmap=plt.cm.Reds)
    plot_grid_map(Rprime, "Reward (IRL)", cmap=plt.cm.Blues)
    plt.show()


if __name__ == '__main__':
    test_gridworld_maxent_irl()
