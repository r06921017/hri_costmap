from collections import namedtuple

import numpy as np


transition = namedtuple('transition', ('obs', 'act', 'next_obs', 'rew'))

class Dataset:
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
