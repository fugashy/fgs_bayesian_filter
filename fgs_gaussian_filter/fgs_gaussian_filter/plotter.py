# -*- coding: utf-8 -*-
import math

import matplotlib.pyplot as plt
import numpy as np


class EKFHistory(object):
    def __init__(self):
        self._x_gt = np.zeros((4, 1))
        self._x_est = np.zeros((4, 1))
        self._x_noised = np.zeros((4, 1))
        self._z = np.zeros((2, 1))

    def plot(self, x_gt, x_est, x_noised, z, x_cov_est):
        self._x_gt = np.hstack((self._x_gt, x_gt))
        self._x_est = np.hstack((self._x_est, x_est))
        self._x_noised = np.hstack((self._x_noised, x_noised))
        self._z = np.hstack((self._z, z))

        plt.cla()
        plt.plot(self._z[0, :], self._z[1, :], '.g')
        plt.plot(self._x_gt[0, :].flatten(),
                 self._x_gt[1, :].flatten(), '-b')
        plt.plot(self._x_noised[0, :].flatten(),
                 self._x_noised[1, :].flatten(), '-k')
        plt.plot(self._x_est[0, :].flatten(),
                 self._x_est[1, :].flatten(), '-r')
        self.plot_covariance_ellipse(x_est, x_cov_est)
        plt.axis('equal')
        plt.grid(True)
        plt.pause(0.001)

    def plot_covariance_ellipse(self, x_est, x_cov_est):  # pragma: no cover
        Pxy = x_cov_est[0:2, 0:2]
        eigval, eigvec = np.linalg.eig(Pxy)

        if eigval[0] >= eigval[1]:
            bigind = 0
            smallind = 1
        else:
            bigind = 1
            smallind = 0

        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        a = math.sqrt(eigval[bigind])
        b = math.sqrt(eigval[smallind])
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
        R = np.array([[math.cos(angle), math.sin(angle)],
                      [-math.sin(angle), math.cos(angle)]])
        fx = R@(np.array([x, y]))
        px = np.array(fx[0, :] + x_est[0, 0]).flatten()
        py = np.array(fx[1, :] + x_est[1, 0]).flatten()
        plt.plot(px, py, '--r')
