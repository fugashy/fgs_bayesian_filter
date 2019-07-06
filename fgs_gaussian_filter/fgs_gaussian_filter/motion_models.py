# -*- coding: utf-8 -*-
from math import cos, sin

import numpy as np


def create(config):
    if config['type'] == 'circle2d':
        dt = config['dt']
        return Circle2D(dt)
    else:
        raise NotImplementedError('{} is not a type of motion model'.format(
            config['type']))


class Circle2D(object):
    u"""xy平面上の円運動モデル

    与えられた状態・操作からその時どの状態になるかを計算する
    """

    def __init__(self, dt):
        u"""運動モデルとそのヤコビアン，分散を用意"""
        self._dt = dt
        # 状態を取り出すための行列
        # 特に操作を加えなかった場合の成分を取り出すために使う
        f_mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]
            ])
        self._extract_static_x = lambda x: f_mat @ x

        # 操作を加えた場合に状態へ与える影響を取り出す行列
        self._extract_command_x = lambda x, u: np.array(
            [
                [cos(x[2, 0]) * self._dt, 0.0],
                [sin(x[2, 0]) * self._dt, 0.0],
                [0.0, self._dt],
                [1.0, 0.0]
            ]) @ u

        # 運動モデルのヤコビアン
        dx_dyaw = lambda x, u: -u[0, 0] * sin(x[2, 0]) * self._dt
        dx_dv = lambda x, u: cos(x[2, 0]) * self._dt
        dy_dyaw = lambda x, u: u[0, 0] * cos(x[2, 0]) * self._dt
        dy_dv = lambda x, u: sin(x[2, 0]) * self._dt
        self._calc_motion_jacob = lambda x, u: np.array(
            [
                [1.0, 0.0, dx_dyaw(x, u), dx_dv(x, u)],
                [0.0, 1.0, dy_dyaw(x, u), dy_dv(x, u)],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ])

        # 運動モデルの分散
        self._cov = np.diag(
            [
                0.1,
                0.1,
                np.deg2rad(1.0),
                1.0
            ])**2

    def calc_next_motion(self, x, u):
        u"""設定時間分先の状態を計算する"""
        return self._extract_static_x(x) + self._extract_command_x(x, u)

    def calc_motion_jacob(self, x, u):
        u"""設定時間先近傍の振る舞いを近似したヤコビ行列を計算する"""
        return self._calc_motion_jacob(x, u)

    @property
    def shape(self):
        return (4, 1)

    @property
    def cov(self):
        u"""運動モデルの分散を取り出す"""
        return self._cov
