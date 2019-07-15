# -*- coding: utf-8 -*-
import math

import matplotlib.pyplot as plt

import numpy as np


def create(config, bayesian_filter):
    if config['type'] == 'simple':
        return SimplePlotter(bayesian_filter)
    elif config['type'] == 'with_landmark':
        return PlotterWithLandMark(bayesian_filter)
    else:
        raise NotImplementedError('{} is not a type of command'.format(
            config['type']))


class Plotter():
    def __init__(self, bayesian_filter):
        self._bayesian_filter = bayesian_filter

    def plot(self, x_gt, x_noised, z):
        raise NotImplementedError('To developers, inherit this class')

    def _plot_covariance_ellipse(self, x_est, x_cov_est):  # pragma: no cover
        Pxy = x_cov_est[0:2, 0:2]
        eigval, eigvec = np.linalg.eig(Pxy)

        if eigval[0] >= eigval[1]:
            bigind = 0
            smallind = 1
        else:
            bigind = 1
            smallind = 0

        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        try:
            a = math.sqrt(eigval[bigind])
        except ValueError:
            a = 0.0
        try:
            b = math.sqrt(eigval[smallind])
        except ValueError:
            b = 0.0
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
        R = np.array([[math.cos(angle), math.sin(angle)],
                      [-math.sin(angle), math.cos(angle)]])
        fx = R@(np.array([x, y]))
        px = np.array(fx[0, :] + x_est[0, 0]).flatten()
        py = np.array(fx[1, :] + x_est[1, 0]).flatten()
        plt.plot(px, py, '--g')


class SimplePlotter(Plotter):
    def __init__(self, bayesian_filter):
        super().__init__(bayesian_filter)
        self._x_gt_hist = np.zeros((4, 1))
        self._x_est_hist = np.zeros((4, 1))
        self._x_noised_hist = np.zeros((4, 1))
        self._z_hist = np.zeros((2, 1))

    def plot(self, x_gt, x_noised, z):
        self._x_gt_hist = np.hstack((self._x_gt_hist, x_gt))
        self._x_est_hist = np.hstack((self._x_est_hist, self._bayesian_filter.x_est))
        self._x_noised_hist = np.hstack((self._x_noised_hist, x_noised))
        # 観測ができなかった場合を考慮
        if z is not None:
            self._z_hist = np.hstack((self._z_hist, z))

        plt.cla()
        plt.plot(self._z_hist[0, :], self._z_hist[1, :], '.g')
        plt.plot(self._x_gt_hist[0, :].flatten(),
                 self._x_gt_hist[1, :].flatten(), '-b')
        plt.plot(self._x_noised_hist[0, :].flatten(),
                 self._x_noised_hist[1, :].flatten(), '-r')
        plt.plot(self._x_est_hist[0, :].flatten(),
                 self._x_est_hist[1, :].flatten(), '-g')
        self._plot_covariance_ellipse(self._bayesian_filter.x_est, self._bayesian_filter.cov_est)
        plt.axis('equal')
        plt.grid(True)
        plt.pause(0.001)


class PlotterWithLandMark(Plotter):
    def __init__(self, bayesian_filter):
        super().__init__(bayesian_filter)
        self._x_gt = np.zeros((4, 1))
        self._x_est = np.zeros((4, 1))
        self._x_noised = np.zeros((4, 1))
        self._landmarks = bayesian_filter.obs_model.landmarks

    def plot(self, x_gt, x_noised, z):
        self._x_gt = np.hstack((self._x_gt, x_gt))
        self._x_est = np.hstack((self._x_est, self._bayesian_filter.x_est))
        self._x_noised = np.hstack((self._x_noised, x_noised))

        plt.cla()

        if z is not None:
            x_est = self._bayesian_filter.x_est
            for i in range(len(z[:, 0])):
                plt.plot([x_est[0, 0], z[i, 1]], [x_est[1, 0], z[i, 2]], '-g')

        plt.plot(self._landmarks[:, 0], self._landmarks[:, 1], '*g')

        plt.plot(self._x_gt[0, :].flatten(),
                 self._x_gt[1, :].flatten(), '-b')
        plt.plot(self._x_noised[0, :].flatten(),
                 self._x_noised[1, :].flatten(), '-r')
        plt.plot(self._x_est[0, :].flatten(),
                 self._x_est[1, :].flatten(), '-g')
        px, _ = self._bayesian_filter.particles
        plt.plot(px[0, :], px[1, :], ".g")
        self._plot_covariance_ellipse(
            self._bayesian_filter.x_est, self._bayesian_filter.cov_est)
        plt.axis('equal')
        plt.grid(True)
        plt.pause(0.001)
