from abc import ABC, abstractmethod

import numpy as np
import mdptoolbox.mdp as mdp
from scipy.special import softmax


class Policy(ABC):
    @abstractmethod
    def get_action(self, obs):
        raise NotImplementedError

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
        assert self.eps >= 0 and self.eps <= 1

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

class BoltzmannPolicy(Policy):
    def __init__(self, action_space, ql_algo, tau=0.1, anneal_rate=1):
        self.action_space = action_space
        self.ql_algo = ql_algo
        self.tau = tau
        self.anneal_rate = anneal_rate

        assert isinstance(self.ql_algo, mdp.QLearning)

        self.Q = ql_algo.Q
        self.t = self.tau

    def get_action(self, obs):
        act_prob = softmax(self.Q[obs] / self.t)
        act = np.random.choice(self.action_space.n, p=act_prob)

        self.t *= self.anneal_rate
        return act

    def reset(self):
        self.t = self.tau
