#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np

if __name__ == '__main__':
    import commands
    import motion_models
    import observation_models
    import noise_models
    import extended_kalman_filter
    import plotter
else:
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
    noise_model = noise_models.SampleNoiseExposure()
    ekf = extended_kalman_filter.ExtendedKalmanFilter(motion_model, obs_model)
    plot = plotter.EKFHistory()

    try:
        while time < SIM_TIME:
            time += DT

            # 操作を与えて真値を更新
            u = command_model.command()
            x_gt = motion_model.calc_next_motion(x_gt, u)

            # そこで観測
            z = obs_model.observe_at(x_gt)

            # ノイズを与える
            u_noised, z_noised = noise_model.expose(u, z)

            # ノイズ操作を用いて運動モデルを更新する(こいつはこれの繰り返し)
            x_noised = motion_model.calc_next_motion(x_noised, u_noised)

            # EKF
            x_est, x_cov_est = ekf.bayesian_update(u_noised, z_noised)

            plot.plot(x_gt, x_est, x_noised, z_noised, x_cov_est)
    except KeyboardInterrupt:
        print('Interrupted by user')


if __name__ == '__main__':
    ekf_sample()
