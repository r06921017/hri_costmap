import numpy as np


class MaxEntIRL:
    '''Maxumum Entropy Inverse Reinforcement Learning.'''

    def __init__(self,
                 env,
                 dataset,
                 phi,
                 max_iter=100):
        self.env = env
        self.dataset = dataset
        self.phi = phi
        self.max_iter = max_iter

    def train(self):
        init_prob = self._compute_initial_state_probability()
        feat_exp = self._compute_feature_expectations()

        for i in range(len(self.max_iter)):
            feature_rewards = self.phi.dot(weights)
            svf_exp = _compute_state_visitation_frequencies()
            loss = feature_rewards - self.phi.dot(svf_exp)
            # TODO: optimize
            # TODO: convergence critereon

    def _compute_initial_state_probability(self):
        prob = np.zeros(self.env.observation_space().n, dtype=np.float)
        for t in self.dataset:
            prob[t[0].obs] += 1
        prob = prob / len(self.dataset)
        return prob

    def _compute_feature_expectations(self):
        pass

    def _compute_state_visitation_frequencies(self):
        pass
