# -*- coding: utf-8 -*-
import numpy as np


def create(config):
    if config['type'] == 'velocity_and_yaw_const':
        vel = config['vel']
        yaw_rate = config['yaw_rate']
        std_dev = config['std_dev']
        return VelocityAndYawConst(vel, yaw_rate, std_dev)
    else:
        raise NotImplementedError('{} is not a type of command'.format(
            config['type']))


class VelocityAndYawConst():
    u"""一定の速度・角速度を与えるクラス."""

    def __init__(self, v, yaw, std_dev):
        u"""
        xyの速度は同じ

        それと角度
        """
        self.vel = v
        self.yaw = yaw
        self._command = np.array(
            [
                [v],
                [yaw]
            ])

        self._cov = np.diag(std_dev)**2

    def command(self, with_noise=False):
        u"""操作を実施する"""
        if with_noise:
            return self._command + \
                self._cov @ np.random.randn(self._command.shape[0], 1)
        else:
            return self._command

    @property
    def cov(self):
        return self._cov
