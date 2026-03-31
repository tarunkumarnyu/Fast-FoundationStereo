#!/bin/bash
export ROS_DOMAIN_ID=17
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source /opt/ros/humble/setup.bash
cd /home/race6/falcon/src/Fast-FoundationStereo
exec python3 -u scripts/live_ffs_unified.py --ros-args \
  -p engine_dir:=/home/race6/falcon/src/Fast-FoundationStereo/engines/480x320_8iter_best \
  -p ns:=/race6/cam1
