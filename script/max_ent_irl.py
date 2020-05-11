import numpy as np


class MaxEntIRL:
    '''
    Maxumum Entropy Inverse Reinforcement Learning.

    Based on Ziebart, et al. (2008).

    Influence from Maximilian Luz's implementation at
    https://github.com/qzed/irl-maxent.git
    '''

    def __init__(self,
                 observation_space,
                 action_space,
                 transition,
                 goal_states,
                 dataset,
                 feature_map,
                 max_iter=10,
                 eps=0.01,
                 lr=0.1,
                 anneal_rate=0.9):
        '''
        :param observation_space: gym.spaces.Discrete object
        :param action_space: gym.spaces.Discrete object
        :param transition: transition function, implemented as matrix
            T[a][s, s'] -> prob
        :param goal_states: list of goal states
        :param dataset: Dataset object of demonstration trajectories
        :param feature_map: feature map of size SxD
        :param max_iter: maximum number of training iterations
        :param eps: stopping critereon
        :param lr: learning rate of optimizer
        '''
        # env information
        self.observation_space = observation_space
        self.action_space = action_space
        self.transition = transition
        self.goal_states = goal_states

        # irl information
        self.dataset = dataset
        self.feature_map = feature_map

        # hyperparameters
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
            # TODO: move optimization to separate class
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
        weights = np.ones((self.feature_map.shape[1], 1), dtype=float)
        return weights

    def _initial_state_probability(self):
        prob = np.zeros(self.observation_space.n, dtype=float)
        for traj in self.dataset:
            prob[traj[0].obs] += 1
        prob /= len(self.dataset)
        return prob

    def _init_feature_expectations(self):
        feat_exp = np.zeros(self.feature_map.shape[1], dtype=float)
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
        exp_feat_rew = np.exp(feat_rew).flatten()
        s_part = np.zeros((S,), dtype=np.longdouble)
        a_part = np.zeros((S, A), dtype=np.longdouble)
        s_part[self.goal_states] = 1.

        for i in range(2 * H):
            a_part[:] = 0.
            for a in range(A):
                a_part[:, a] = T[a].dot(s_part) * exp_feat_rew
            s_part = np.sum(a_part, axis=1)

        # Local action probability
        local_a_prob = a_part / s_part.reshape((-1, 1))

        # Forward pass: compute svf
        svf = np.zeros((S,), dtype=float)
        savf = np.zeros((S, A), dtype=float)

        for i in range(1, 2 * H):
            for a in range(A):
                savf[:, a] = T[a].T.dot(local_a_prob[:, a] * svf)
            svf = init_prob + np.sum(savf, axis=1)

        return svf
