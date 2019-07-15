# -*- coding: utf-8 -*-
import math
from copy import deepcopy

import numpy as np

from scipy import linalg as LA


def create(config, command_model, motion_model, obs_model):
    if config['type'] == 'ekf':
        return ExtendedKalmanFilter(command_model, motion_model, obs_model)
    elif config['type'] == 'ukf':
        alpha = config['alpha']
        beta = config['beta']
        kappa = config['kappa']
        return UnscentedKalmanFilter(
            alpha, beta, kappa, command_model, motion_model, obs_model)
    elif config['type'] == 'pf':
        p_num = config['p_num']
        p_num_resample = config['p_resample']
        if p_num < p_num_resample:
            raise ValueError('p num should be less than p num for resampling')
        resample_threshold = config['resample_threshold']
        return ParticleFilter(
            p_num, p_num_resample, resample_threshold,
            command_model, motion_model, obs_model)
    else:
        raise NotImplementedError('{} is not a type of gaussian filter'.format(
            config['type']))


class BayesianFilter():
    def __init__(self, command_model, motion_model, obs_model):
        self._command_model = command_model
        self._motion_model = motion_model
        self._obs_model = obs_model

    def bayesian_update(self, u, z=None):
        u"""ベイズフィルタ更新

        操作は必須，観測はできない場合があるとする
        """
        raise NotImplementedError('To developers, inherit this class')


class ExtendedKalmanFilter(BayesianFilter):
    u"""EKFの更新を行う."""

    def __init__(self, command_model, motion_model, obs_model):
        u"""推定値の初期化と各種モデルの保持."""
        super().__init__(command_model, motion_model, obs_model)
        self._x_est = np.zeros(motion_model.shape)
        self._cov_est = np.eye(motion_model.shape[0])

    def bayesian_update(self, u, z=None):
        u"""ベイズフィルタ更新を行う."""
        # 持っている状態を事前のものとする
        x_pre = deepcopy(self._x_est)
        cov_pre = deepcopy(self._cov_est)

        # 予測predict
        # 前回の状態を元に操作を与えた場合の状態
        x_pred = self._motion_model.calc_next_motion(x_pre, u)
        # その時の運動モデルのヤコビ行列
        x_jacob = self._motion_model.calc_motion_jacob(x_pred, u)
        # 分散の予測
        cov_pred = x_jacob @ cov_pre @ x_jacob.T + self._motion_model.cov

        # 観測できない場合は予測のみ
        # 分散は増えていく
        if z is None:
            self._x_est = x_pred
            self._cov_est = cov_pred
            return self._x_est, self._cov_est

        # 更新update
        # 計測モデルのヤコビ行列
        z_jacob = self._obs_model.calc_jacob(x_pred)
        # 予測した状態における計測
        z_pred = self._obs_model.observe_at(x_pred)
        # ?
        S = z_jacob @ cov_pred @ z_jacob.T + self._obs_model.cov
        # ?
        K = cov_pred @ z_jacob.T @ LA.inv(S)
        # 実際の計測との差
        z_diff = z - z_pred
        # 状態の修正
        self._x_est = x_pred + K @ z_diff
        # 状態のばらつきの修正
        self._cov_est = (np.eye(len(self._x_est)) - K @ z_jacob) @ cov_pred

        return self._x_est, self._cov_est


class UnscentedKalmanFilter(BayesianFilter):
    u"""アンセッテッドカルマンフィルタ

    さっぱりわかってない
    """

    def __init__(self, alpha, beta, kappa, command_model, motion_model, obs_model):
        u"""初期化

        パラメータの計算と推定する状態等の次元決定
        各種モデルの保持
        """
        super().__init__(command_model, motion_model, obs_model)
        self._wm = 0.
        self._wc = 0.
        self._gamma = 0.
        self._init_param(motion_model.shape[0], alpha, beta, kappa)

        self._x_est = np.zeros(motion_model.shape)
        self._cov_est = np.eye(motion_model.shape[0])

    def bayesian_update(self, u, z):
        x_pre = deepcopy(self._x_est)
        cov_pre = deepcopy(self._cov_est)

        #  Predict
        sigma = self._generate_sigma_points(x_pre, cov_pre)
        sigma = self._predict_sigma_motion(sigma, u)
        x_pred = (self._wm @ sigma.T).T
        cov_pred = self._calc_sigma_cov(
            x_pred, sigma, self._motion_model.cov)

        if z is None:
            self._x_est = x_pred
            self._cov_est = cov_pred
            return self._x_est, self._cov_est

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

    def _generate_sigma_points(self, x, cov):
        sigma = deepcopy(x)
        cov_squared = LA.sqrtm(cov)
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


class ParticleFilter(BayesianFilter):
    def __init__(
            self, p_num, p_num_resample, resample_threshold,
            command_model, motion_model, obs_model):
        super().__init__(command_model, motion_model, obs_model)
        self._p_num = p_num
        self._p_num_resample = p_num_resample
        self._resample_threshold = resample_threshold

        self._px = np.zeros((self._motion_model.shape[0], p_num))
        self._pw = np.zeros((1, p_num)) + 1.0 / p_num

        # 状態の初期化
        # 分散共分散については過去のものを考慮せずに毎回計算するので保持しない
        self._x_est = np.zeros(self._motion_model.shape)

    def bayesian_update(self, u, z=None):
        # 設定されたパーティクル数だけ実施する
        for ip in range(self._p_num):
            # 操作モデルをランダムに揺さぶり，そのときの状態を計算する
            # すなわち，パーティクルのばらまきをする
            ud = u + self._command_model.cov @ np.random.randn(u.shape[0], 1)
            p = np.array([self._px[:, ip]]).T
            p = self._motion_model.calc_next_motion(p, ud)

            # パーティクルから観測
            zp = self._obs_model.observe_at(p)

            # そのパーティクルの尤度を更新
            pw = self._pw[0, ip]

            # 観測がない場合は尤度の計算ができない
            # よって重みが更新されない
            if z is not None:
                for iz in range(len(zp[:, 0])):
                    try:
                        # 事前に観測したときの距離との差
                        # パーティクルから観測すると観測できなかった点も含まれる場合がある
                        # そのためアクセスエラーはスキップする
                        likelihood = self._obs_model.gauss_likelihood(zp[iz], z[iz])
                    except IndexError:
                        continue
                    pw *= likelihood

            # パーティクルの更新
            self._px[:, ip] = p[:, 0]
            self._pw[0, ip] = pw

        # 正規化
        self._pw /= self._pw.sum()

        # 更新
        # 加重総和
        # 正規化済みなのでスケールは変わらない
        self._x_est = self._px @ self._pw.T
        cov_est = self._calc_cov()

        # リサンプリングする
        # ぶっちゃけわからん
        self._resampling()

        return self._x_est, cov_est

    @property
    def particles(self):
        return self._px, self._pw

    def _calc_cov(self):
        cov = np.zeros(
            (self._motion_model.shape[0], self._motion_model.shape[0]))
        for i in range(self._px.shape[1]):
            dx = (self._px[:, i] - self._x_est)
            cov += self._pw[0, i] * (dx @ dx.T)

        # 原典(python robotics)では速度のばらつきを無視している
        # Why
        cov[:, -1] = 0.0
        cov[-1, :] = 0.0

        return cov

    def _resampling(self):
        # 重みの2乗和
        # 軽く計算したら，尤度分布の尖り具合と相関があるっぽい
        neff = 1.0 / (self._pw @ self._pw.T)[0, 0]

        if neff < self._resample_threshold:
            # 重みの累積配列
            wcum = np.cumsum(self._pw)
            # TODO(fugashy)なんらかのアルゴリズムでリサンプリングすべきIDを得ているが
            # わからない
            base = np.cumsum(
                self._pw * 0.0 + 1 / self._p_num) - 1 / self._p_num
            resample_id = base + np.random.rand(base.shape[0]) / self._p_num

            inds = []
            ind = 0
            for ip in range(self._p_num):
                while resample_id[ip] > wcum[ind]:
                    ind += 1
                inds.append(ind)
            self._px = self._px[:, inds]
            self._pw = np.zeros((1, self._p_num)) + 1.0 / self._p_num
