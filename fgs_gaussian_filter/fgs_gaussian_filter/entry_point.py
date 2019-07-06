#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from sys import argv, exit

import yaml

import offline_simulator


def main(args):
    if len(args) != 2:
        print('usage: python3 fgs_gaussian_filter.entry_point PATH_TO_CONFIG_YAML')
        return -1

    f = open(args[1], 'r')
    config = yaml.load(f, Loader=yaml.FullLoader)

    sim = offline_simulator.OfflineSimulator(config, as_node=False)
    sim.run()

    return 0


if __name__ == '__main__':
    exit(main(argv))
