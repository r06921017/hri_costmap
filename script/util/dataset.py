from collections import namedtuple

import numpy as np
from tqdm import trange

# MDP transition
transition = namedtuple('transition', ('obs', 'act', 'next_obs', 'rew'))

class Dataset:
    ''' Container class for finite mdp trajectories.'''
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.data = []

    def append(self, trajectory):
        t = trajectory[:self.maxlen]
        self.data.append(t)

    def __iter__(self):
        for t in self.data:
            yield t

    def __len__(self):
        return len(self.data)


def collect_trajectories(policy,
                         env,
                         num_trajectories,
                         maxlen,
                         rendered=False):
    dataset = Dataset(maxlen=maxlen)

    for n in range(num_trajectories):
        t = []
        obs, rew, done, info = env.reset()
        policy.reset()

        for i in range(maxlen):
            act = policy.get_action(obs)
            next_obs, rew, done, info = env.step(act)
            t.append(transition(
                obs=obs, act=act, next_obs=next_obs, rew=rew))

            if done:
                break
            obs = next_obs
            if rendered:
                time.sleep(0.2)

        dataset.append(t)

    return dataset

