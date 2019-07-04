# -*- coding: utf-8 -*-
import numpy as np


class SampleNoiseExposure(object):
    u"""PythonRoboticsと同様のノイズ付加を行うクラス"""

    def __init__(self):
        u"""モデルの登録とノイズ付加関数の用意"""
        self._expose_noise_to_com = lambda u: \
            u + ((np.diag([1.0, np.deg2rad(30.0)])**2) @ np.random.randn(2, 1))

        self._expose_noise_to_obs = lambda z: \
            z + (np.diag([0.5, 0.5])**2) @ np.random.randn(2, 1)

    def expose(self, u, z):
        u"""操作と観測にノイズに曝す"""
        u_noised = self._expose_noise_to_com(u)
        z_noised = self._expose_noise_to_obs(z)

        return u_noised, z_noised
