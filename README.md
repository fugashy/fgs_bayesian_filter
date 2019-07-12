# fgs_bayesian_filter

A package that provide samples attempting localization based on bayesian filters.

# Implemented bayesian filters

- Gaussian Filters

  - Extended Kalman Filter

  - Enscented Kalman Filter

- Non-parametric Filters

  - Particle Filter

# My environment

- OS: MacOS Mojave 10.14.5 (18F203)

# How to use

- As ROS2

  ```bash
  ros2 run fgs_bayesian_filter start __params:=PATH_TO_THIS_PKG/config/bayesian_filter.yaml
  ```

- As python3 script

  ```bash
  cd PATH_TO_THIS_PKG
  ./fgs_bayesian_filter/entry_point.py
  ```

![localization_sample](https://github.com/fugashy/fgs_bayesian_filter/blob/doc/images/ekf.gif)

- Blue arrows are groundtruth.
- Red arrows are dead-reckoning(only odometry).
- Green arrows are poses estimated by Bayesian Filter.

# Special Thanks

- [確率ロボティクス](https://book.mynavi.jp/ec/products/detail/id=37337)

- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)

  ```
  The MIT License (MIT)

  Copyright (c) 2016 Atsushi Sakai

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ```

