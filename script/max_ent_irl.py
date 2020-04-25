from itertools import product

import matplotlib.pyplot as plt
from mdptoolbox.mdp import ValueIteration
import numpy as np

from dataset import Dataset, transition
from grid_world import GridWorld


class MaxEntIRL:
    '''Maxumum Entropy Inverse Reinforcement Learning.'''

    def __init__(self,
                 env,
                 dataset,
                 feature_map,
                 init_weight_var=1.,
                 max_iter=10,
                 lr=0.1,
                 eps=0.01):
        self.env = env
        self.dataset = dataset
        self.feature_map = feature_map
        self.init_weight_var = init_weight_var
        self.max_iter = max_iter
        self.eps = 0.01
        self.lr = lr

    def train(self):
        weights = self._init_feature_weights()
        self.w = weights
        prev_weights = np.empty_like(weights)
        init_prob = self._initial_state_probability()
        feat_exp = self._init_feature_expectations()

        for i in range(self.max_iter):
            # calculate loss gradient
            feature_rewards = self.feature_map.dot(weights)
            svf_exp = self._state_visitation_frequencies(
                feature_rewards, init_prob)
            grad = feat_exp - self.feature_map.T.dot(svf_exp)
            
            # optimize
            np.copyto(dst=prev_weights, src=weights)
            weights *= np.exp(self.lr * grad).reshape((-1, 1))

            # convergence critereon
            delta = np.max(np.abs(weights - prev_weights))
            print "iteration {0}, delta={1}".format(i, delta)
            if delta < self.eps:
                break

        return self.feature_map.dot(weights)

    def _init_feature_weights(self):
        D = self.feature_map.shape[1]
        weights = np.random.normal(
            loc=0., scale=np.sqrt(self.init_weight_var), size=(D, 1))
        return weights

    def _initial_state_probability(self):
        prob = np.zeros(self.env.observation_space().n, dtype=np.float)
        for traj in self.dataset:
            prob[traj[0].obs] += 1
        prob = prob / len(self.dataset)
        return prob

    def _init_feature_expectations(self):
        feat_exp = np.zeros(self.feature_map.shape[1], dtype=np.float)
        for traj in self.dataset:
            for trans in traj:
                feat_exp += self.feature_map[trans.obs, :]
        feat_exp /= self.dataset.maxlen
        return feat_exp

    def _state_visitation_frequencies(self, feat_rew, init_prob):
        S = self.env.observation_space().n
        A = self.env.action_space().n
        H = self.dataset.maxlen

        # Backward pass: compute state partition
        s_part = np.zeros(S, dtype=np.float)
        a_part = np.zeros((S, A), dtype=np.float)
        s_part[env.goal_state] = 1.

        for i in range(2 * H):
            a_part[:] = 0.

            for s, a, sprime in product(range(S), range(A), range(S)):
                a_part[s, a] += (self.env.T[a][s, sprime] *
                                 np.exp(feat_rew[s]) *
                                 s_part[sprime])
            s_part = np.sum(a_part, axis=1)

        # Local action probability
        local_a_prob = a_part / s_part.reshape((-1, 1))

        # Forward pass: compute svf
        stvf = np.zeros((S, 2 * H), dtype=np.float)
        stvf[:, 0] = init_prob
        nongoal_states = [s for s in range(S) if s != self.env.goal_state]

        for i in range(1, 2 * H):
            for s, a, sprime in product(nongoal_states, range(A), range(S)):
                stvf[sprime, i] += (stvf[s, i-1] *
                                   local_a_prob[s, a] *
                                   self.env.T[a][s, sprime])

        svf = np.sum(stvf, axis=1)
        return svf

if __name__ == '__main__':
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
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2, subplot_kw={'xticklabels': [], 'yticklabels': []})
    
    ax1.set_title("Reward (Ground truth)")
    ax1.matshow(R, cmap=plt.cm.Reds)

    ax2.set_title("Value Function")
    ax2.matshow(V, cmap=plt.cm.Blues)
    plt.show()
    # import pdb; pdb.set_trace()

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
    # import pdb; pdb.set_trace()

    # phi
    phi = np.eye(env.observation_space().n, dtype=np.float)

    # IRL
    me_irl = MaxEntIRL(
        env,
        dataset,
        phi,
        max_iter=20)
    Rprime = me_irl.train()

    # import pdb; pdb.set_trace()