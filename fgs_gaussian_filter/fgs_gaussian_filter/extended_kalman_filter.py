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
        # 前回の状態を元に操作を与えた場合の状態
        x_pred = self.motion_model.calc_next_motion(x_pre, u)
        # その時の運動モデルのヤコビ行列
        x_jacob = self.motion_model.calc_motion_jacob(x_pred, u)
        # 分散の予測
        cov_pred = x_jacob @ cov_pre @ x_jacob.T + self.motion_model.cov

        # 更新update
        # 計測モデルのヤコビ行列
        z_jacob = self.obs_model.calc_jacob(x_pred)
        # 予測した状態における計測
        z_pred = self.obs_model.observe_at(x_pred)
        # ?
        S = z_jacob @ cov_pred @ z_jacob.T + self.obs_model.cov
        # ?
        K = cov_pred @ z_jacob.T @ np.linalg.inv(S)
        # 実際の計測との差
        z_diff = z - z_pred
        # 状態の修正
        self.x_est = x_pred + K @ z_diff
        # 状態のばらつきの修正
        self.cov_est = (np.eye(len(self.x_est)) - K @ z_jacob) @ cov_pred

        return self.x_est, self.cov_est
