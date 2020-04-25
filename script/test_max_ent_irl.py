import matplotlib.pyplot as plt
from mdptoolbox.mdp import ValueIteration
import numpy as np

from dataset import Dataset, transition
from grid_world import GridWorld
from max_ent_irl import MaxEntIRL
from util.plotting import plot_grid_map


def test_gridworld_maxent_irl():
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
        render=False
    )

    # dataset
    gamma = 0.9
    vi = ValueIteration(env.T, env.R, gamma)
    vi.run()
    pi = vi.policy

    R = env.R.reshape((N, N)).T
    V = np.asarray(vi.V).reshape((N, N)).T

    # TODO: stochastic policy generation + sampling
    dataset = Dataset(maxlen=8)
    t = []
    obs, rew, done, info = env.reset()
    while not done:
        act = pi[obs]
        next_obs, rew, done, info = env.step(act)
        t.append(transition(obs=obs, act=act, next_obs=next_obs, rew=rew))
        obs = next_obs
    dataset.append(t)

    # phi
    phi = np.eye(env.observation_space().n, dtype=np.float)

    # IRL
    me_irl = MaxEntIRL(
        env,
        dataset,
        phi,
        max_iter=10)
    Rprime = me_irl.train()
    Rprime = Rprime.reshape((N, N))

    # plot results
    plot_grid_map(R, "Reward (Ground Truth)", cmap=plt.cm.Reds)
    plot_grid_map(Rprime, "Reward (IRL)", cmap=plt.cm.Blues)
    plt.show()


if __name__ == '__main__':
    test_gridworld_maxent_irl()
