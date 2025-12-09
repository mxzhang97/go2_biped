#!/bin/bash
echo "Setting up ros2 environment, swapping in cyclonedds middleware"
source /opt/ros/humble/setup.bash
source $HOME/go2_biped/go2_deploy_ros2/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
                            <NetworkInterface name="lo" priority="default" multicast="default" />
                        </Interfaces></General></Domain></CycloneDDS>'


