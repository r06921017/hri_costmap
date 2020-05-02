from abc import ABC, abstractmethod

import numpy as np
import mdptoolbox.mdp as mdp


class Policy(ABC):
    @abstractmethod
    def get_action(self, obs):
        raise NotImplementedError

    def reset(self):
        pass

class RandomPolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        act = np.random.choice(self.action_space.n)
        return act

class GreedyPolicy(Policy):
    def __init__(self, action_space, mdp_algo):
        self.action_space = action_space
        self.mdp_algo = mdp_algo

        assert (isinstance(self.mdp_algo, mdp.ValueIteration) or
                isinstance(self.mdp_algo, mdp.QLearning) or
                isinstance(self.mdp_algo, mdp.PolicyIteration))

        self.policy = np.asarray(mdp_algo.policy)

    def get_action(self, obs):
        act = self.policy[obs]
        return act

class EpsGreedyPolicy(Policy):
    def __init__(self, action_space, mdp_algo, epsilon=0.2, anneal_rate=1):
        self.action_space = action_space
        self.mdp_algo = mdp_algo
        self.epsilon = epsilon
        self.anneal_rate = 1

        assert (isinstance(self.mdp_algo, mdp.ValueIteration) or
                isinstance(self.mdp_algo, mdp.QLearning) or
                isinstance(self.mdp_algo, mdp.PolicyIteration))
        assert self.epsilon >= 0 and self.epsilon <= 1

        self.policy = np.asarray(mdp_algo.policy)
        self.eps = self.epsilon

    def get_action(self, obs):
        if np.random.uniform() < self.eps:
            act = np.random.choice(self.action_space.n)
        else:
            act = self.policy[obs]

        self.eps *= self.anneal_rate
        return act

    def reset(self):
        self.eps = self.epsilon

class StochasticGreedyPolicy(Policy):
    def __init__(self, action_space, mdp_algo, transition):
        self.action_space = action_space
        self.mdp_algo = mdp_algo
        self.transition = transition

        assert (isinstance(self.mdp_algo, mdp.ValueIteration) or
                isinstance(self.mdp_algo, mdp.QLearning) or
                isinstance(self.mdp_algo, mdp.PolicyIteration))

        self.V = np.asarray(self.mdp_algo.V)
        self._generate_policy()

    def _generate_policy(self):
        self.pi = np.zeros(
            (self.V.size, self.action_space.n), dtype=np.float)
        for s in range(self.V.size):
            act_value_exp = []
            for a in range(self.action_space.n):
                sprime_dist = self.transition[a][s].toarray().flatten()
                act_value = np.sum(self.V * sprime_dist)
                act_value_exp.append(act_value)
            act_value_exp = np.array(act_value_exp)
            value_exp = act_value_exp - np.min(act_value_exp)

            if np.sum(value_exp) == 0:
                act_prob = np.ones(self.action_space.n) / self.action_space.n
            else:
                act_prob = value_exp / np.sum(value_exp)

            self.pi[s, :] = act_prob


    def get_action(self, obs):
        act_prob = self.pi[obs, :]
        act = np.random.choice(self.action_space.n, p=act_prob)

        return act

class BoltzmannPolicy(Policy):
    def __init__(self, action_space, ql_algo, tau=0.1, anneal_rate=1):
        self.action_space = action_space
        self.ql_algo = ql_algo
        self.tau = tau
        self.anneal_rate = anneal_rate

        assert isinstance(self.ql_algo, mdp.QLearning)

        self.Q = np.copy(ql_algo.Q)
        self.t = self.tau

    def get_action(self, obs):
        act_exp = np.exp(self.Q[obs] - np.max(self.Q[obs]))
        act_prob = act_exp / np.sum(act_exp)
        act = np.random.choice(self.action_space.n, p=act_prob)

        self.t *= self.anneal_rate
        return act

    def reset(self):
        self.t = self.tau