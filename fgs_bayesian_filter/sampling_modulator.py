# -*- coding: utf-8 -*-
import numpy as np
import time


def create(config):
    if config['type'] == 'uniform_dist':
        prob = config['prob']
        return UniformDistributedSampler(prob)
    elif config['type'] == 'time_based_interval':
        interval_sec = config['interval_sec']
        return TimeBasedIntervalSampler(interval_sec)
    else:
        raise NotImplementedError('{} is not a type of gaussian filter'.format(
            config['type']))


class Sampler():
    def is_valid(self):
        raise NotImplementedError('To developers, inherit this class')


class UniformDistributedSampler(Sampler):
    def __init__(self, prob=1.0):
        super().__init__()
        self._prob = prob

    def is_valid(self):
        return np.random.uniform() >= 1.0 - self._prob


class TimeBasedIntervalSampler(Sampler):
    def __init__(self, interval_sec):
        super().__init__()
        self._interval_sec = interval_sec
        self._previous_sec = None

    def is_valid(self):
        if self._previous_sec is None:
            self._previous_sec = time.time()
            return True

        current_sec = time.time()
        if current_sec - self._previous_sec > self._interval_sec:
            self._previous_sec = current_sec
            return True
        return False
