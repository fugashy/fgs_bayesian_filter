# -*- coding: utf-8 -*-
import numpy as np
import time


class LineMotion(object):
    u"""
    1次元直線運動モデル
    """
    def __init__(self, u, s):
        u"""
        Args:
            u: 平均速度[m/s] (float)
            s: 速度標準偏差[m/s] (float)
        """
        self.u = np.array([[u]]).T
        self.cov = np.array([[s**2]])

        self.previous_time = None

    def get_covariance(self):
        return self.cov

    def get_odometry(self):
        if self.previous_time is None:
            self.previous_time = time.time()
        dt = time.time() - self.previous_time

        return self.u * dt
