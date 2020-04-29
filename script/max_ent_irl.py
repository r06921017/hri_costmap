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
                 eps=0.01,
                 lr=0.1,
                 anneal_rate=0.9):
        '''
        :param observation_space: gym.spaces.Discrete object
        :param action_space: gym.spaces.Discrete object
        :param transition: transition function, implemented as a callable
            T(s, a, s') -> prob or matrix T[a][s, s'] -> prob
        :param goal_states: list of goal states
        :param dataset: Dataset object of demonstration trajectories
        :param feature_map: feature map of size SxD
        :param init_weight_var: variance of learned weights initialization
        :param max_iter: maximum number of training iterations
        :param eps: stopping critereon
        :param lr: learning rate of optimizer
        '''
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
        self.eps = eps
        self.lr = lr
        self.anneal_rate = anneal_rate

    def train(self):
        self.weights = self._init_feature_weights()
        self.prev_weights = np.empty_like(self.weights)
        self.init_prob = self._initial_state_probability()
        self.feat_exp = self._init_feature_expectations()

        for i in range(self.max_iter):
            # calculate loss gradient
            feature_rewards = self.feature_map.dot(self.weights)
            svf_exp = self._state_visitation_frequencies(
                feature_rewards, self.init_prob)
            grad = self.feat_exp - self.feature_map.T.dot(svf_exp)
            
            # optimize
            # TODO: break out optimization to separate class
            np.copyto(dst=self.prev_weights, src=self.weights)
            self.weights *= np.exp(self.lr * grad).reshape((-1, 1))
            self.lr *= self.anneal_rate

            # convergence critereon
            delta = np.max(np.abs(self.weights - self.prev_weights))
            print("iteration {0}, delta={1}".format(i, delta))
            if delta < self.eps:
                break

        return self.feature_map.dot(self.weights)

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
        return prob

    def _init_feature_expectations(self):
        feat_exp = np.zeros(self.feature_map.shape[1], dtype=np.float)
        for traj in self.dataset:
            for trans in traj:
                feat_exp += self.feature_map[trans.obs, :]
        feat_exp /= len(self.dataset)
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
