# -*- coding: utf-8 -*-
import numpy as np


class VelocityAndYawConst():
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
        return self._command
