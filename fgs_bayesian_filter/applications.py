# -*- coding: utf-8 -*-
import time
import numpy as np

import rclpy

import yaml

from fgs_bayesian_filter import (
    bayesian_filters,
    command_models,
    motion_models,
    observation_models,
    ros_utils
)

from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped


class EstimatedStatePublisherNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('estimated_state_publisher')

        self.declare_parameter(name='config_path', value='')
        config_path = self.get_parameter('config_path').value

        f = open(config_path, 'r')
        config = yaml.load(f, Loader=yaml.FullLoader)

        self._motion_model = motion_models.create(config['motion_model'])
        self._obs_model = observation_models.create(config['observation_model'])
        self._command_model = command_models.create(config['command_model'])
        self._filter = bayesian_filters.create(
            config['bayesian_filter'], self._command_model, self._motion_model, self._obs_model)

        self._x_gt = np.zeros(self._motion_model.shape)
        self._x_noised = np.zeros(self._motion_model.shape)  # 放っておくとどうなるか

        # Publishers
        # Current states
        # NOTE(fugashy) At 20190709, PoseWithCovariance cannot be visualized on RViz2
        self._pub_x_est = self.create_publisher(PoseWithCovarianceStamped, 'estimated_state', 1)
        self._pub_x_gt = self.create_publisher(PoseWithCovarianceStamped, 'groundtruth_state', 1)
        # History
        self._pub_x_est_array = self.create_publisher(PoseArray, 'estimated_state_array', 1)
        self._pub_x_gt_array = self.create_publisher(PoseArray, 'groundtruth_state_array', 1)
        self._pub_x_dr_array = self.create_publisher(PoseArray, 'dead_reckoning_state_array', 1)
        self._x_est_array_msg = PoseArray()
        self._x_est_array_msg.header.frame_id = 'map'
        self._x_gt_array_msg = PoseArray()
        self._x_gt_array_msg.header.frame_id = 'map'
        self._x_dr_array_msg = PoseArray()
        self._x_dr_array_msg.header.frame_id = 'map'

    def run(self):
        try:
            while True:
                u = self._command_model.command(with_noise=False)
                self._x_gt = self._motion_model.calc_next_motion(self._x_gt, u)

                u_noised = self._command_model.command(with_noise=True)
                z_noised = self._obs_model.observe_at(self._x_gt, with_noise=True)

                self._x_noised = self._motion_model.calc_next_motion(
                    self._x_noised, u_noised)

                x_est, x_cov_est = self._filter.bayesian_update(u_noised, z_noised)

                # 出版
                # 推定した状態をmessageに変換する
                x_est_msg = ros_utils.IntegrateEstimated2DPoseAsROSMessage(x_est, x_cov_est)
                self._x_est_array_msg.poses.append(x_est_msg.pose.pose)
                x_gt_msg = ros_utils.ConvertEstimated2DPoseAsROSMessage(self._x_gt)
                self._x_gt_array_msg.poses.append(x_gt_msg.pose.pose)
                x_dr_msg = ros_utils.ConvertEstimated2DPoseAsROSMessage(self._x_noised)
                self._x_dr_array_msg.poses.append(x_dr_msg.pose.pose)

                self._pub_x_est.publish(x_est_msg)
                self._pub_x_est_array.publish(self._x_est_array_msg)
                self._pub_x_gt.publish(x_gt_msg)
                self._pub_x_gt_array.publish(self._x_gt_array_msg)
                self._pub_x_dr_array.publish(self._x_dr_array_msg)

                time.sleep(0.3)
        except KeyboardInterrupt:
            print('Interrupted by user')
