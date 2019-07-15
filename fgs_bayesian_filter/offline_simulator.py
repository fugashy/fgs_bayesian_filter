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
            from fgs_bayesian_filter import (
                bayesian_filters,
                command_models,
                motion_models,
                observation_models,
                plotters,
                sampling_modulator
            )
        else:
            import bayesian_filters
            import command_models
            import motion_models
            import observation_models
            import plotters
            import sampling_modulator

        self._motion_model = motion_models.create(config['motion_model'])
        self._obs_model = observation_models.create(config['observation_model'])
        self._command_model = command_models.create(config['command_model'])
        self._sampler = None
        if 'sampler' in config:
            self._sampler = sampling_modulator.create(config['sampler'])
        self._filter = bayesian_filters.create(
            config['bayesian_filter'], self._command_model, self._motion_model, self._obs_model)
        self._plotter = plotters.create(config['plotter'], self._filter)

        self._x_gt = np.zeros(self._motion_model.shape)
        self._x_noised = np.zeros(self._motion_model.shape)  # 放っておくとどうなるか

    def run(self):
        try:
            while True:
                # 操作を与えて真値を更新
                u = self._command_model.command(with_noise=False)
                self._x_gt = self._motion_model.calc_next_motion(self._x_gt, u)

                # 操作と観測
                u_noised = self._command_model.command(with_noise=True)
                z_noised = None
                if self._sampler is None or self._sampler.is_valid():
                    z_noised = self._obs_model.observe_at(self._x_gt, with_noise=True)

                # ノイズを受けた状態・操作を用いて運動モデルを更新する(こいつはこれの繰り返し)
                self._x_noised = self._motion_model.calc_next_motion(
                    self._x_noised, u_noised)

                # Localization
                self._filter.bayesian_update(u_noised, z_noised)

                # 可視化
                self._plotter.plot(self._x_gt, self._x_noised, z_noised)
        except KeyboardInterrupt:
            print('Interrupted by user')
