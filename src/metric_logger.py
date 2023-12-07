import torch
from collections import defaultdict, deque
import numpy as np

class MetricLogger:
    def __init__(self):
        self.values = defaultdict(SmoothedValue)

    def update(self, **kargs):
        for key, value in kargs.items():
            self.values[key].update(value)

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, item):
        self.values[key].update(item)

class SmoothedValue:
    """
    Record the last several values, and return summaries
    """

    def __init__(self, maxsize=20):
        self.values = deque(maxlen=maxsize)
        self.count = 0
        self.sum = 0.0

    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.values.append(value)
        self.count += 1
        self.sum += value

    @property
    def median(self):
        return np.median(np.array(self.values))

    @property
    def avg(self):
        return np.mean(self.values)

    @property
    def global_avg(self):
        return self.sum / self.count
