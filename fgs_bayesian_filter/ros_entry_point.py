# -*- coding: utf-8 -*-
from sys import exit

import rclpy
from rclpy.node import Node

import yaml

# TODO(fugashy) create factory of application
from fgs_bayesian_filter import applications


def main(args=None):
    rclpy.init(args=args)

    node = applications.EstimatedStatePublisherNode()
    node.run()

    exit(0)
