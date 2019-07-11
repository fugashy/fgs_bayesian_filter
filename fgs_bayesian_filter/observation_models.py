u"""Observation models such as GPS."""
# -*- coding: utf-8 -*-
import math

import numpy as np


def create(config):
    if config['type'] == 'gps_xy':
        std_dev = config['std_dev']
        if len(std_dev) != 2:
            raise ValueError('Input std dev is not valid')
        return GPSObservation(std_dev)
    elif config['type'] == 'rfid_xyd':
        id_list = config['id_list']
        max_range = config['max_range']
        std_dev = config['std_dev']
        if len(std_dev) != 2:
            raise ValueError('Input std dev is not valid')
        return RFIDXYD(id_list, std_dev, max_range)
    if config['type'] == 'gps_xyY':
        std_dev = config['std_dev']
        if len(std_dev) != 3:
            raise ValueError('Input std dev is not valid')
        return GPSAndDirection(std_dev)
    else:
        raise NotImplementedError('{} is not a type of observation_model'.format(
            config['type']))


class GPSObservation():
    u"""位置のみを観測するクラス.

    GPSを想定している
    """

    def __init__(self, std_dev):
        u"""
        観測行列の定義を行う.

        状態ベクトルから，観測に対応する部分を取り出す役割をする
        """
        self._obs_mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ])

        self._obs_jacob_mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ])

        self._cov = np.diag(std_dev)**2

    def observe_at(self, x, with_noise=False):
        u"""与えられた状態をもとに観測する."""
        if with_noise:
            return self._obs_mat @ x + self._cov @ np.random.randn(2, 1)
        else:
            return self._obs_mat @ x

    def calc_jacob(self, x):
        u"""ヤコビアンを返す

        状態を与える必要はないが，他クラスを意識して与えることにする
        """
        return self._obs_jacob_mat

    @property
    def cov(self):
        return self._cov


class RFIDXYD():
    u"""無線的なセンサを使って2Dの点と距離を計測するクラス"""

    def __init__(self, id_list, std_dev, max_range):
        u"""初期化

        検知するランドマークの2D位置のリストを保持する
        [[x1, y1], [x2, y2], ..., [xn, yn]]
        """
        self._id_list = np.array(id_list)
        self._max_range = max_range

        # 状態からxy位置を取り出すための行列
        self.obs_mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ])

        self._cov = np.diag(std_dev)**2

    def observe_at(self, x, with_noise=False):
        observed_pose = self.obs_mat @ x

        # 距離計算して，計測範囲内のRFIDのみ抽出する
        # [distance, x, y]の3要素
        z = np.zeros((0, 3))
        for i in range(len(self._id_list)):
            if with_noise:
                dx = (observed_pose[0] - self._id_list[i, 0]) + \
                    np.random.randn() * self._cov[0, 0]
                dy = (observed_pose[1] - self._id_list[i, 1]) + \
                    np.random.randn() * self._cov[1, 1]
            else:
                dx = observed_pose[0] - self._id_list[i, 0]
                dy = observed_pose[1] - self._id_list[i, 1]
            d = math.sqrt(dx**2 + dy**2)
            if d <= self._max_range:
                z = np.vstack((z, np.array(
                    [d, self._id_list[i, 0], self._id_list[i, 1]])))
        return z

    def gauss_likelihood(self, z1, z2):
        u"""互いの距離が近いほど，確からしいとする"""
        dz = z1[0] - z2[0]
        # 距離についての正規分布の式で評価
        # 距離が近いほど値が大きくなる
        likelihood = \
            1.0 / math.sqrt(2.0 * math.pi * self._cov[0, 0]) * \
            math.exp(-dz**2 / (2.0 * self._cov[0, 0]))
        return likelihood

    @property
    def cov(self):
        return self._cov

    @property
    def landmark(self):
        return self._id_list


class GPSAndDirection():
    def __init__(self, std_dev):
        self._obs_mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ])
        self._obs_jacob_mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ])
        self._cov = np.diag(std_dev)**2

    def observe_at(self, x, with_noise=False):
        if with_noise:
            return self._obs_mat @ x + self._cov @ np.random.randn(3, 1)
        else:
            return self._obs_mat @ x

    def calc_jacob(self, x):
        u"""ヤコビアンを返す

        状態を与える必要はないが，他クラスを意識して与えることにする
        """
        return self._obs_jacob_mat

    @property
    def cov(self):
        return self._cov
