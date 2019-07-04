# -*- coding: utf-8 -*-
import numpy as np

class SampleNoiseModel(object):
    def __init__(self, motion_model, observation_model):
        self.command_noise = lambda: np.dot(
                np.diag([1.0, np.deg2rad(30.0)])**2,
                np.random.randn(2, 1)
            )

        self.obs_noise = lambda: np.dot(
                np.diag([0.5, 0.5])**2,
                np.random.randn(2, 1)
            )

        self.motion_model = motion_model
        self.observation_model = observation_model


    def noise_expose(self, state_gt, x_dr, u):
        # 真値の更新
        state_gt = self.motion_model.calc_next_dt_motion(state_gt, u)

        # 真値をベースに観測し，ノイズを与える
        z_noised = self.observation_model.observe_at(state_gt) + self.obs_noise()

        # 操作にノイズを与える
        u_noised = u + self.command_noise()

        # 状態にノイズを与える
        x_noised = self.motion_model.calc_next_dt_motion(x_dr, u_noised)

        return state_gt, z_noised, x_noised, u_noised
