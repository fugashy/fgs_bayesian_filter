# -*- coding: utf-8 -*-
from sys import exit

import rclpy
from rclpy.node import Node

import yaml

# TODO(fugashy) create factory of application
from fgs_gaussian_filter import offline_simulator


def main(args=None):
    rclpy.init(args=args)

    node = Node('bayesian_filter')

    node.declare_parameter(name='config_path', value='')
    config_path = node.get_parameter('config_path').value

    f = open(config_path, 'r')
    config = yaml.load(f, Loader=yaml.FullLoader)

    sim = offline_simulator.OfflineSimulator(config, as_node=True)
    sim.run()

    exit(0)
