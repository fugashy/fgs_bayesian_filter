# -*- coding: utf-8 -*-
import numpy as np


class VelocityAndYawConst():
    u"""一定の速度・角速度を与えるクラス."""

    def __init__(self, v, yaw):
        u"""
        xyの速度は同じ

        それと角度
        """
        self.vel = v
        self.yaw = yaw
        self._command = np.array(
            [
                [v],
                [yaw]
            ])

    def command(self):
        u"""操作を実施する"""
        return self._command
