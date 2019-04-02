cmake_minimum_required(VERSION 2.8.3)
project(fgs_bayesian)
find_package(catkin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  fgs_opt_data_storage
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS fgs_opt_data_storage
)
