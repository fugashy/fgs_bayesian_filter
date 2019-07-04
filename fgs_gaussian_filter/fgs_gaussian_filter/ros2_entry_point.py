# -*- coding: utf-8 -*-
import numpy as np


from fgs_gaussian_filter import (
    commands,
    motion_models,
    observation_models,
    noise_models,
    extended_kalman_filter,
    plotter
)

SIM_TIME = 120.0
DT = 0.1
VEL_POS = 1.0
VEL_YAW = 0.1


def ekf_sample(args=None):
    print('ekf start !')

    time = 0.0

    # [x, y, yaw, v]
    x_est = np.zeros((4, 1))
    x_gt = np.zeros((4, 1))
    x_cov_est = np.eye(4)
    x_noised = np.zeros((4, 1))  # 放っておくとどうなるか

    motion_model = motion_models.Circle2D(DT)
    command_model = commands.VelocityAndYawConst(VEL_POS, VEL_YAW)
    obs_model = observation_models.GPSObservation()
    data = noise_models.SampleNoiseExposure(motion_model, obs_model)
    ekf = extended_kalman_filter.ExtendedKalmanFilter(motion_model, obs_model)
    plot = plotter.EKFHistory()

    try:
        while time < SIM_TIME:
            time += DT

            u = command_model.command()

            x_gt, z_noised, x_noised, u_noised = data.expose(
                x_gt, x_noised, u)

            x_est, x_cov_est = ekf.bayesian_update(u_noised, z_noised)

            plot.plot(x_gt, x_est, x_noised, z_noised, x_cov_est)
    except KeyboardInterrupt:
        print('Interrupted by user')
