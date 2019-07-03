# -*- coding: utf-8 -*-
import numpy as np


from fgs_gaussian_filter import (
    commands, noise_models, extended_kalman_filter, models, plotter
)

SIM_TIME = 120.0
DT = 0.1
VEL_POS = 1.0
VEL_YAW = 0.1

def ekf_sample(args=None):
    print('ekf start !')

    time = 0.0

    # [x, y, yaw, v]
    state_estimated = np.zeros((4, 1))
    state_groundtruth = np.zeros((4, 1))
    state_covariance = np.eye(4)
    # ?
    state_noised = np.zeros((4, 1))

    motion_model = models.Circle2D(DT)
    command_model = commands.VelocityAndYawConst(VEL_POS, VEL_YAW)
    obs_model = models.GPSObservation()
    data = noise_models.SampleNoiseModel(motion_model, obs_model)
    ekf = extended_kalman_filter.ExtendedKalmanFilter(motion_model, obs_model)
    plot = plotter.EKFHistory()

    while time < SIM_TIME:
        time += DT

        command = command_model.command()
        print(command)

        state_groundtruth, obs_noised, state_noised, command_noised = data.noise_expose(
                state_groundtruth, state_noised, command)

        state_estimate, state_covariance = ekf.bayesian_update(
                command_noised, obs_noised)

        plot.plot(state_groundtruth, state_estimated, state_dead_noised, obs_noised)
