# -*- coding: utf-8 -*-
u"""EKF."""
import math
from copy import deepcopy

import numpy as np

import scipy


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


class UnscentedKalmanFilter():
    u"""アンセッテッドカルマンフィルタ

    さっぱりわかってない
    """

    def __init__(self, alpha, beta, kappa, x, motion_model, obs_model):
        u"""初期化

        パラメータの計算と推定する状態等の次元決定
        各種モデルの保持
        """
        x_len = len(x)
        self._wm = 0.
        self._wc = 0.
        self._gamma = 0.
        self._init_param(x_len, alpha, beta, kappa)

        self._x_est = np.zeros(x.shape)
        self._cov_est = np.eye(x_len)

        self._motion_model = motion_model
        self._obs_model = obs_model

    def bayesian_update(self, u, z):
        x_pre = deepcopy(self._x_est)
        cov_pre = deepcopy(self._cov_est)

        #  Predict
        sigma = self._generate_sigma_points(x_pre, cov_pre)
        sigma = self._predict_sigma_motion(sigma, u)
        x_pred = (self._wm @ sigma.T).T
        cov_pred = self._calc_sigma_cov(
            x_pred, sigma, self._wc, self._motion_model.cov)

        #  Update
        z_pred = self._obs_model.observe_at(x_pred)
        z_diff = z - z_pred
        sigma = self._generate_sigma_points(x_pred, cov_pred)
        zb = (self._wm @ sigma.T).T
        z_sigma = self._predict_sigma_observation(sigma)
        st = self._calc_sigma_cov(zb, z_sigma, self._obs_model.cov)
        Pxz = self._calc_pxz(sigma, x_pred, z_sigma, zb)
        K = Pxz @ np.linalg.inv(st)
        self._x_est = x_pred + K @ z_diff
        self._cov_est = cov_pred - K @ st @ K.T

        return self._x_est, self._cov_est

    def _init_param(self, x_len, alpha, beta, kappa):
        lamb = alpha ** 2 * (x_len + kappa) - x_len
        # calculate weights
        wm = [lamb / (lamb + x_len)]
        wc = [(lamb / (lamb + x_len)) + (1. - alpha**2 + beta)]
        for i in range(2 * x_len):
            wm.append(1.0 / (2 * (x_len + lamb)))
            wc.append(1.0 / (2 * (x_len + lamb)))
        self._gamma = math.sqrt(x_len + lamb)

        self._wm = np.array([wm])
        self._wc = np.array([wc])

    def generate_sigma_points(self, x, cov):
        sigma = deepcopy(x)
        cov_squared = scipy.linalg.sqrtm(cov)
        n = len(x[:, 0])

        for i in range(n):
            sigma = np.hstack((sigma, x + self._gamma * cov_squared[:, i:i + 1]))

        for i in range(n):
            sigma = np.hstack((sigma, x - self._gamma * cov_squared[:, i:i + 1]))

        return sigma

    def _predict_sigma_motion(self, sigma, u):
        for i in range(sigma.shape[1]):
            sigma[:, i:i + 1] = self._motion_model.calc_next_motion(
                sigma[:, i:i + 1], u)

        return sigma

    def _predict_sigma_observation(self, sigma):
        for i in range(sigma.shape[1]):
            sigma[0:2, i] = self._obs_model.observe_at(sigma[:, i])

        sigma = sigma[0:2, :]

        return sigma

    def _calc_sigma_cov(self, x, sigma, cov):
        d = sigma - x[0:sigma.shape[0]]
        cov_out = deepcopy(cov)
        for i in range(sigma.shape[1]):
            cov_out = cov_out + self._wc[0, i] * d[:, i:i + 1] @ d[:, i:i + 1].T
        return cov_out

    def _calc_pxz(self, sigma, x, z_sigma, zb):
        dx = sigma - x
        dz = z_sigma - zb[0:2]
        p = np.zeros((dx.shape[0], dz.shape[0]))

        for i in range(sigma.shape[1]):
            p = p + self._wc[0, i] * dx[:, i:i + 1] @ dz[:, i:i + 1].T

        return p
