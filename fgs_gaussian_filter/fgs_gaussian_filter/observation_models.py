# -*- coding: utf-8 -*-
import numpy as np


class LineObservation(object):
    u"""与えられた入力に対して、
    定められた正規分布に基づいた観測を返す
    """
    def __init__(self, s):
        u"""
        Args:
            s: 観測のばらつき[m/s](float)
        """
        self.s = s
        self.cov = np.array([[s**2]])

    def get_covariance(self):
        return self.cov

    def get_observation(self, x):
        u"""
        Args:
            x: 基準点[m](float)
        """
        return np.random.normal(x, self.s)
