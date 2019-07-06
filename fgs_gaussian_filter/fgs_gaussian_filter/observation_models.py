u"""Observation models such as GPS."""
# -*- coding: utf-8 -*-
import numpy as np


def create(config):
    if config['type'] == 'gps_xy':
        return GPSObservation()
    else:
        raise NotImplementedError('{} is not a type of observation_model'.format(
            config['type']))


class GPSObservation():
    u"""位置のみを観測するクラス.

    GPSを想定している
    """

    def __init__(self):
        u"""
        観測行列の定義を行う.

        状態ベクトルから，観測に対応する部分を取り出す役割をする
        """
        self.obs_mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ])

        self.obs_jacob_mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ])

        self.cov = np.diag([1.0, 1.0])**2

    def observe_at(self, x):
        u"""与えられた状態をもとに観測する."""
        return self.obs_mat @ x

    def calc_jacob(self, x):
        u"""ヤコビアンを返す

        状態を与える必要はないが，他クラスを意識して与えることにする
        """
        return self.obs_jacob_mat

    def cov(self):
        return self.cov
