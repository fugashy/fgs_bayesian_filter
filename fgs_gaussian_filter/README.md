# fgs_gaussian_filter

自己位置推定のためのガウシアンフィルタ群  
クラス設計を重視して，流れをつかみやすくすることを意識  

# Implemented gaussian filters

- Extended Kalman Filter
- Enscented Kalman Filter

# My environment

- OS: MacOS Mojave 10.14.5 (18F203)

# How to use

- As ROS2

  ```bash
  ros2 run fgs_gaussian_filter start __params:=PATH_TO_THIS_PKG/config/gaussian_filter.yaml
  ```

- As python3 script

  ```bash
  cd PATH_TO_THIS_PKG
  ./fgs_gaussian_filter/entry_point.py
  ```
