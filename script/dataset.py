from collections import namedtuple

import numpy as np


transition = namedtuple('transition', ('obs', 'act', 'new_obs'))

class Dataset:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.data = []

    def append(self, trajectory):
        t = trajectory[:self.maxlen]
        self.data.append(t)

    def __iter__(self):
        for t in data:
            yield t

    def __len__(self):
        return len(self.data)
