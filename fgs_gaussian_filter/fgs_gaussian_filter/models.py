# -*- coding: utf-8 -*-
import math

import numpy as np


class Circle2D(object):
    def __init__(self, dt):
        self.dt = dt
        self.state_mat = lambda: np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]
            ])

        # x[2, 0] = yaw
        self.command_mat = lambda x: np.array(
            [
                [self.dt * math.cos(x[2, 0]), 0.0       ],
                [self.dt * math.sin(x[2, 0]), 0.0       ],
                [0.0,                         self.dt],
                [1.0,                         0.0]
            ])

        dx_dyaw = lambda x, u: -u[0, 0] * self.dt * math.sin(x[2, 0])
        dx_dv = lambda x, u: self.dt * math.cos(x[2, 0])
        dy_dyaw = lambda x, u: u[0, 0] * self.dt * math.cos(x[2, 0])
        dy_dv = lambda x, u: self.dt * math.sin(x[2, 0])
        self.state_jacob_mat = lambda x, u: np.array(
            [
                [1.0, 0.0, dx_dyaw(x, u), dx_dv(x, u)],
                [0.0, 1.0, dy_dyaw(x, u), dy_dv(x, u)],
                [0.0, 0.0, 1.0,           0.0        ],
                [0.0, 0.0, 0.0,           1.0        ]
            ])

        self.cov = np.diag(
            [
                0.1,
                0.1,
                np.deg2rad(1.0),
                1.0
            ])**2

    def calc_next_dt_motion(self, x_t, u_t):
        u"""
        設定時間分先の状態を計算する
        """
        x_pred = np.dot(self.state_mat(), x_t) + np.dot(self.command_mat(x_t), u_t)
        print(x_pred)
        return x_pred

    def calc_jacob_of_state(self, x_pred, u_t):
        u"""
        設定時間先近傍の振る舞いを近似したヤコビ行列を計算する
        """
        return self.state_jacob_mat(x_pred, u_t)


class GPSObservation():
    def __init__(self):
        u"""
        観測行列の定義を行う.

        状態ベクトルから，観測に対応する部分を取り出す役割をする
        """
        self.observation_mat = lambda: np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ])

        self.observation_jacob_mat = lambda: np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ])

        self.cov = np.diag([1.0, 1.0])**2

    def observe_at(self, x_t):
        return self.observation_mat() @ x_t
