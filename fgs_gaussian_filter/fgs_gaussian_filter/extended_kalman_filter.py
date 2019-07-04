# -*- coding: utf-8 -*-
u"""EKF."""
from copy import deepcopy

import numpy as np


class ExtendedKalmanFilter():
    u"""EKFの更新を行う."""

    def __init__(self, motion_model, obs_model):
        u"""推定値の初期化と各種モデルの保持."""
        self.x_est = np.zeros((4, 1))
        self.cov_est = np.eye(4)

        self.motion_model = motion_model
        self.obs_model = obs_model

    def bayesian_update(self, u, z):
        u"""ベイズフィルタ更新を行う."""
        # 持っている状態を事前のものとする
        x_pre = deepcopy(self.x_est)
        cov_pre = deepcopy(self.cov_est)

        # 予測predict
        x_pred = self.motion_model.calc_next_dt_motion(x_pre, u)
        x_jacob = self.motion_model.calc_jacob_of_state(x_pred, u)
        cov_pred = x_jacob @ cov_pre @ x_jacob.T + self.motion_model.cov

        # 更新update
        z_jacob = self.obs_model.observation_jacob_mat()
        z_pred = self.obs_model.observe_at(x_pred)
        y = z - z_pred
        S = z_jacob @ cov_pred @ z_jacob.T + self.obs_model.cov
        K = cov_pred @ z_jacob.T @ np.linalg.inv(S)
        self.x_est = x_pred + K @ y
        self.cov_est = (np.eye(len(self.x_est)) - K @ z_jacob) @ cov_pred

        return self.x_est, self.cov_est
