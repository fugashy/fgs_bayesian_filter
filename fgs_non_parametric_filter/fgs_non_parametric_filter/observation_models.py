# -*- coding: utf-8 -*-
import math

import numpy as np


def create(config):
    if config['type'] == 'rfid_xyd':
        id_list = config['id_list']
        max_range = config['max_range']
        std_dev = config['std_dev']
        if len(std_dev) != 2:
            raise ValueError('Input std dev is not valid')
        return RFIDXYD(id_list, std_dev, max_range)
    else:
        raise NotImplementedError('{} is not a type of observation_model'.format(
            config['type']))


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

    def observe_at(self, x):
        observed_pose = self.obs_mat @ x

        # 距離計算して，計測範囲内のRFIDのみ抽出する
        # [distance, x, y]の3要素
        z = np.zeros((0, 3))
        for i in range(len(self._id_list)):
            dx = observed_pose[0] - self._id_list[i, 0]
            dy = observed_pose[1] - self._id_list[i, 1]
            d = math.sqrt(dx**2 + dy**2)
            if d <= self._max_range:
                z = np.vstack((z, np.array(
                    [d, self._id_list[i, 0], self._id_list[i, 1]])))
        return z

    @property
    def cov(self):
        return self._cov

    @property
    def landmark(self):
        return self._id_list
