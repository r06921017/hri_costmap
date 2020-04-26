from itertools import product

import matplotlib.pyplot as plt
from mdptoolbox.mdp import ValueIteration
import numpy as np

from dataset import Dataset, transition
from grid_world import GridWorld


class MaxEntIRL:
    '''Maxumum Entropy Inverse Reinforcement Learning.'''

    def __init__(self,
                 observation_space,
                 action_space,
                 transition,
                 goal_states,
                 dataset,
                 feature_map,
                 init_weight_var=1.,
                 max_iter=10,
                 lr=0.1,
                 eps=0.01):
        # env information
        self.observation_space = observation_space
        self.action_space = action_space
        if callable(transition):
            self.transition = transition
        else:
            self.transition = lambda s, a, sprime: transition[a][s, sprime]
        self.goal_states = goal_states

        # irl information
        self.dataset = dataset
        self.feature_map = feature_map

        # hyperparameters
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
            print("iteration {0}, delta={1}".format(i, delta))
            if delta < self.eps:
                break

        return self.feature_map.dot(weights)

    def _init_feature_weights(self):
        D = self.feature_map.shape[1]
        weights = np.random.normal(
            loc=0., scale=np.sqrt(self.init_weight_var), size=(D, 1))
        return weights

    def _initial_state_probability(self):
        prob = np.zeros(self.observation_space.n, dtype=np.float)
        for traj in self.dataset:
            prob[traj[0].obs] += 1
        prob = prob / len(self.dataset)
        import pdb; pdb.set_trace()
        return prob

    def _init_feature_expectations(self):
        feat_exp = np.zeros(self.feature_map.shape[1], dtype=np.float)
        for traj in self.dataset:
            for trans in traj:
                feat_exp += self.feature_map[trans.obs, :]
        feat_exp /= self.dataset.maxlen
        return feat_exp

    def _state_visitation_frequencies(self, feat_rew, init_prob):
        S = self.observation_space.n
        A = self.action_space.n
        H = self.dataset.maxlen
        T = self.transition

        # Backward pass: compute state partition
        s_part = np.zeros(S, dtype=np.float)
        a_part = np.zeros((S, A), dtype=np.float)
        s_part[self.goal_states] = 1.

        for i in range(2 * H):
            a_part[:] = 0.

            for s, a, sprime in product(range(S), range(A), range(S)):
                a_part[s, a] += (T(s, a, sprime) *
                                 np.exp(feat_rew[s]) *
                                 s_part[sprime])
            s_part = np.sum(a_part, axis=1)

        # Local action probability
        local_a_prob = a_part / s_part.reshape((-1, 1))

        # Forward pass: compute svf
        nongoal_states = [s for s in range(S) if s not in self.goal_states]
        stvf = np.zeros((S, 2 * H), dtype=np.float)
        stvf[:, 0] = init_prob

        for i in range(1, 2 * H):
            for s, a, sprime in product(nongoal_states, range(A), range(S)):
                stvf[sprime, i] += (stvf[s, i-1] *
                                   local_a_prob[s, a] *
                                   T(s, a, sprime))

        svf = np.sum(stvf, axis=1)
        return svf
