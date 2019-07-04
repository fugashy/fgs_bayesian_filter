# -*- coding: utf-8 -*-
import numpy as np


# TODO: 真値の更新等をここで行う必要はないので切り出す
class SampleNoiseExposure(object):
    u"""PythonRoboticsと同様のノイズ付加を行うクラス"""

    def __init__(self, motion_model, obs_model):
        u"""モデルの登録とノイズ付加関数の用意"""
        self.expose_noise_to_com = lambda u: \
            u + ((np.diag([1.0, np.deg2rad(30.0)])**2) @ np.random.randn(2, 1))

        self.expose_noise_to_obs = lambda z: \
            z + (np.diag([0.5, 0.5])**2) @ np.random.randn(2, 1)

        self.motion_model = motion_model
        self.obs_model = obs_model

    def expose(self, xgt, x, u):
        u"""ノイズに曝す"""
        # 真値の更新
        xgt = self.motion_model.calc_next_motion(xgt, u)

        # 真値をベースに観測し，ノイズを与える
        z_noised = self.expose_noise_to_obs(self.obs_model.observe_at(xgt))

        # 操作にノイズを与える
        u_noised = self.expose_noise_to_com(u)

        # 状態にノイズを与える
        x_noised = self.motion_model.calc_next_motion(x, u_noised)

        return xgt, z_noised, x_noised, u_noised
