# -*- coding: utf-8 -*-
from math import cos, sin
from geometry_msgs.msg import PoseWithCovarianceStamped


def IntegrateEstimated2DPoseAsROSMessage(x_est, x_cov_est, frame_id='map'):
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = frame_id
    msg.pose.pose.position.x = x_est[0, 0]
    msg.pose.pose.position.y = x_est[1, 0]
    msg.pose.pose.position.z = 0.0

    cos_yaw = cos(x_est[2, 0] / 2.0);
    sin_yaw = sin(x_est[2, 0] / 2.0);

    msg.pose.pose.orientation.x = 0.0
    msg.pose.pose.orientation.y = 0.0
    msg.pose.pose.orientation.z = sin_yaw
    msg.pose.pose.orientation.w = cos_yaw

    msg.pose.covariance[0] = x_cov_est[0, 0]
    msg.pose.covariance[1] = x_cov_est[0, 1]
    msg.pose.covariance[6] = x_cov_est[1, 0]
    msg.pose.covariance[7] = x_cov_est[1, 1]

    return msg


def ConvertEstimated2DPoseAsROSMessage(x_est, frame_id='map'):
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = frame_id
    msg.pose.pose.position.x = x_est[0, 0]
    msg.pose.pose.position.y = x_est[1, 0]
    msg.pose.pose.position.z = 0.0

    cos_yaw = cos(x_est[2, 0] / 2.0);
    sin_yaw = sin(x_est[2, 0] / 2.0);

    msg.pose.pose.orientation.x = 0.0
    msg.pose.pose.orientation.y = 0.0
    msg.pose.pose.orientation.z = sin_yaw
    msg.pose.pose.orientation.w = cos_yaw

    return msg
