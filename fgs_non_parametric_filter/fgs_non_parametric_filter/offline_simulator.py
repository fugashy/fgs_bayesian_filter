# -*- coding: utf-8 -*-

import numpy as np


class OfflineSimulator():
    u"""データ生成しながら逐次更新を行うクラス"""

    def __init__(self, config, as_node=False):
        u"""初期化

        各種モデル
        更新していく状態等の定義
        """
        if as_node:
            from fgs_gaussian_filter import (
                command_models,
                non_parametric_filters,
                motion_models,
                noise_models,
                observation_models,
                plotters
            )
        else:
            import command_models
            import non_parametric_filters
            import motion_models
            import noise_models
            import observation_models
            import plotters

        self._motion_model = motion_models.create(config['motion_model'])
        self._obs_model = observation_models.create(config['observation_model'])
        self._command = command_models.create(config['command_model'])
        self._noise_model = noise_models.create(
            config['noise_model'], self._command, self._obs_model)
        self._filter = non_parametric_filters.create(
            config['filter'], self._command, self._motion_model, self._obs_model)
        # TODO(fugashy) rename as general name
        self._plotter = plotters.PlotterWithLandMark(self._obs_model)

        self._x_est = np.zeros(self._motion_model.shape)
        self._x_gt = np.zeros(self._motion_model.shape)
        self._x_cov_est = np.eye(self._motion_model.shape[0])
        self._x_noised = np.zeros(self._motion_model.shape)  # 放っておくとどうなるか

    def run(self):
        try:
            while True:
                # 操作を与えて真値を更新
                u = self._command.command()
                self._x_gt = self._motion_model.calc_next_motion(self._x_gt, u)

                # そこで観測
                z = self._obs_model.observe_at(self._x_gt)

                # ノイズを与える
                u_noised, z_noised = self._noise_model.expose(u, z)

                # ノイズを受けた状態・操作を用いて運動モデルを更新する(こいつはこれの繰り返し)
                self._x_noised = self._motion_model.calc_next_motion(
                    self._x_noised, u_noised)

                # Localization
                x_est, x_cov_est = self._filter.bayesian_update(u_noised, z_noised)
                px, _ = self._filter.particles

                self._plotter.plot(
                    self._x_gt, x_est, self._x_noised, z_noised, x_cov_est, px)
        except KeyboardInterrupt:
            print('Interrupted by user')