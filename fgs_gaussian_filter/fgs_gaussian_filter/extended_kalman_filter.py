# -*- coding: utf-8 -*-
from copy import deepcopy



class ExtendedKalmanFilter(object):
    def __init__(self, motion_model, obs_model):
        self.state_est = None
        self.cov_est = None

        self.motion_model = motion_model
        self.obs_model = obs_model

    def bayesian_update(self, command, obs):
        # 持っている状態を事前のものとする
        state_pre = deepcopy(self.state_est)
        cov_pre = deepcopy(self.cov_est)

        # 予測predict
        state_predict = self.motion_model.calc_next_dt_motion(state_pre, command)
        state_jacob = self.motion_model.calc_jacob_of_state(state_predict, command)
        cov_predict = np.dot(
                np.dot(state_jacob, cov_pre),
                state_jacob.T)

        # 更新update
        obs_predict = self.obs_model.observation_jacob_mat(state_predict)
        y = obs - obs_predict
        S = np.dot(np.dot(obs_predict, cov_predict), obs_predict.T) + self.motion_model.cov
        K = np.dot(
            np.dot(cov_predict, obs_predict.T),
            np.linalg.inv(S))
        self.state_est = state_predict + np.dot(K, y)
        self.cov_est = np.dot(
                (np.eye(len(self.state_est)) - np.dot(K, obs_predict)),
                cov_predict)

        return self.state_est, self.cov_est
