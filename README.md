# Robodoggo (Biped Locomotion + 1D arm tracking)

## Credits:
* [Genesis LR](https://github.com/lupinjia/genesis_lr)
* [Legged Gym](https://github.com/leggedrobotics/legged_gym)
* [Go2 Deploy](https://github.com/lupinjia/go2_deploy)
* [Bipedal Locomotion for Quadrupedal Robots](https://github.com/arclab-hku/bipedal_locomotion_for_quadrupedal_robots)
* [Learning Quadruped Locomotion Policies using Logical Rules](https://arxiv.org/html/2107.10969v3)
* [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2)


## Installation

### Training (`go2_learn`)

**System Requirements:**
*   **OS:** Ubuntu 22.04
*   **GPU Driver:** Nvidia Driver 535
*   **Simulator:** IsaacGym Preview 4
*   **Python:** 3.8

Install IsaacGym

```bash
wget https://developer.nvidia.com/isaac-gym-preview-4 \
    && tar -xf isaac-gym-preview-4 \
    && rm isaac-gym-preview-4
```

Substitute np.float with np.float32 to resolve compatibility

```bash
find isaacgym/python -type f -name "*.py" -exec sed -i 's/np\.float/np.float32/g' {} +
```

Create and activate virtual environment

```bash
python3.8 -m venv venv
source ./venv/bin/activate
```

Install IsaacGym into the virtual environment

```bash
cd isaacgym/python && pip install -e .
```

Clone this repo and install various dependencies from `go2_learn/`.

```bash
cd go2_learn && pip install -e . --extra-index-url https://download.pytorch.org/whl/cu121
```

### Deploy (`go2_deploy_ros2`)

Install ROS2 Humble from here: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

Install dependencies

```bash
sudo apt install ros-humble-rmw-cyclonedds-cpp 
sudo apt install ros-humble-rosidl-generator-dds-idl
sudo apt install libyaml-cpp-dev
```

Install Libtorch

```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip
```
Make sure to use the extracted path in `go2_deploy_ros2/ros2_ws/src/main_controller/CMakeLists.txt`
```bash
set(CMAKE_PREFIX_PATH /home/username/libtorch)
```


## Usage

### Training (`go2_learn`)

To start training, train the teacher policy first

```bash
python legged_gym/scripts/transfer_train_teacher.py --task=go2_biped_rm --headless
```

To playback the policy in the simulator

```bash
python legged_gym/scripts/transfer_play_teacher.py \
    --task=go2_biped_rm \
    --num_envs=4 \
    --teacher_checkpoint={path_to_checkpoint}
```


To start distillation into student policy + train state estimator

```bash
python legged_gym/scripts/transfer_train_student_quad.py \
    --task=go2_biped_rm_student_quad \
    --headless \
    --teacher_checkpoint={path_to_teacher_checkpoint}
```

To playback the policy in the simulator

```bash
python legged_gym/scripts/transfer_play_student_quad.py \
    --task=go2_biped_rm_student_quad \
    --num_envs=4 \
    --student_checkpoint={path_to_student_checkpoint}
```

**Note:** To export a deployable JIT file, modify the `EXPORT` flag at the bottom of the `transfer_play_student_quad.py` file.


To monitor training progress, use tensorboard

```bash
tensorboard --logdir {path_to_logs}
```


### Deploy (`go2_deploy_ros2`)

Move the exported JIT checkpoints to: `go2_deploy_ros2/ros2_ws/src/main_controller/models/`

Edit model paths within the `main_controller/include/biped_controller.hpp` accordingly

To compile/build

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Edit `go2_deploy_ros2/init.sh` to source the correct installed workspace

```bash
source {path}/go2_biped/go2_deploy_ros2/ros2_ws/install/setup.bash
```

Run the init.sh script to setup the ros2 environment with the cycloneDDS middleware and run the launch scripts.

```bash
source ./init.sh
ros2 launch go2_sunrise sunrise.launch.py
```



## Validation in Simulator

It's useful to verify everything is working correctly/as expected before deploying on hardware by utilizing [Unitree Mujoco](https://github.com/unitreerobotics/unitree_mujoco)

**Helpful Configuration**

* In `/simulate/config.yaml`, edit `domain_id:0` to test via ROS2 topics.
* For local PC testing, use `interface: "lo"`.
* For testing with onboard compute, use `ifconfig` to find the correct interface name.
* Modify friction coefficients in `unitree_robots/go2/go2.xml` as appropriate.

**Note:** When editing network interface, ensure that the exported CycloneDDS environment variables in `init.sh` (`NetworkInterface` `name` field) matches your testing setup.









