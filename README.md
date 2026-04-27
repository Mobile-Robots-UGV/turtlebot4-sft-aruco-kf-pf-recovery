
# Smart Follower & Tracker TurtleBot 4 ROS 2

ROS 2 Jazzy hardware implementation of the **Smart Follower & Tracker** project on TurtleBot 4.

This repository implements a real-robot ArUco board following and tracking pipeline using the TurtleBot 4 OAK-D camera, calibrated board pose estimation, selectable KF/PF tracking, short-horizon prediction recovery, LiDAR-based safety checking, and closed-loop velocity control.

The system is designed for the RAS Mobile Robotics project milestone sequence and extends the earlier board-following work toward a more complete target tracking and recovery stack.

---

## Project Context

This repository supports the Smart Follower & Tracker project.

The larger project goal is to build a TurtleBot 4 system that can:

- detect an object of interest,
- estimate its relative pose,
- track it over time,
- follow it safely,
- handle temporary target loss,
- and prepare for future obstacle-aware recovery and planning.

This repository focuses on the hardware tracking and following layer.

---

## What This Repository Contains

```text
    turtlebot4-sft-aruco-kf-pf-recovery/
├── board_pose_ros/
│   ├── board_pose_ros/
│   │   └── board_pose_node.py
│   ├── config/
│   │   ├── board_config.json
│   │   └── camera_calib_oak.npz
│   ├── launch/
│   │   └── board_pose.launch.py
│   ├── package.xml
│   └── setup.py
└── sft_hardware_tracker/
    ├── sft_hardware_tracker/
    │   ├── board_tracker_node.py
    │   └── recovery_follower_node.py
    ├── config/
    │   └── sft_hardware_recovery.yaml
    ├── launch/
    │   └── sft_hardware_recovery.launch.py
    ├── package.xml
    └── setup.py
````

---

## Package Overview

## 1. `board_pose_ros`

This package performs camera-based ArUco board pose estimation.

It subscribes to the TurtleBot 4 OAK-D compressed RGB image stream, detects the configured ArUco board, estimates its 6-DoF pose using camera calibration and known board geometry, and publishes ROS topics for downstream tracking and control.

### Main responsibilities

* subscribe to OAK-D compressed image stream,
* decode camera frames using OpenCV,
* detect ArUco markers,
* match detected marker IDs to the configured board layout,
* estimate board pose using `cv2.solvePnP`,
* publish board visibility,
* publish board pose,
* publish detected marker IDs,
* publish roll/pitch/yaw diagnostics,
* broadcast a TF transform from camera frame to board frame.

### Main outputs

```text
/robot_09/board_pose
/robot_09/board_visible
/robot_09/board_used_ids
/robot_09/board_rpy
/tf
```

---

## 2. `sft_hardware_tracker`

This package adds the tracking, prediction, recovery, and following layer.

It contains two main nodes:

```text
board_tracker_node.py
recovery_follower_node.py
```

### `board_tracker_node.py`

The tracker subscribes to the raw board pose and visibility topics and converts them into a target tracking state.

It supports two selectable estimator backends:

```text
kf = Kalman Filter
pf = Particle Filter
```

The tracker publishes:

```text
/robot_09/tracked_board_pose
/robot_09/tracker_status
/robot_09/predicted_board_path
```

The tracking status can be:

```text
measured
predicted
lost
```

### Tracker modes

| Status      | Meaning                         | Behavior                        |
| ----------- | ------------------------------- | ------------------------------- |
| `measured`  | Board is currently visible      | Use filtered live board pose    |
| `predicted` | Board was recently lost         | Predict target pose using KF/PF |
| `lost`      | Board has been missing too long | Stop trusting prediction        |

---

### `recovery_follower_node.py`

The follower subscribes to the tracked board pose, tracker status, and TurtleBot 4 LiDAR scan.

It publishes velocity commands to:

```text
/robot_09/cmd_vel
```

The follower uses:

* board `z` position for forward/backward distance regulation,
* board `x` position for angular centering,
* tracker status for behavior switching,
* LiDAR front range for safety.

### Follower behavior

| Tracker Status | Follower Behavior                |
| -------------- | -------------------------------- |
| `measured`     | Normal visual following          |
| `predicted`    | Slow cautious predicted tracking |
| `lost`         | Stop robot                       |

The follower also uses `/robot_09/scan` to prevent unsafe forward motion when an obstacle is directly in front of the robot.

---

## System Architecture

```text
TurtleBot 4 OAK-D Camera
        |
        v
board_pose_node
        |
        | /robot_09/board_pose
        | /robot_09/board_visible
        v
board_tracker_node
        |
        | /robot_09/tracked_board_pose
        | /robot_09/tracker_status
        | /robot_09/predicted_board_path
        v
recovery_follower_node
        |
        | /robot_09/cmd_vel
        v
TurtleBot 4 Base
```

With LiDAR safety:

```text
/robot_09/scan
        |
        v
recovery_follower_node
```

---

## Hardware Platform

Tested hardware target:

```text
Robot: TurtleBot 4
Camera: OAK-D
Sensor: TurtleBot 4 LiDAR
Board: Printed ArUco marker board
```

Robot namespace used in this project:

```text
/robot_09
```

---

## Software Stack

```text
Ubuntu 24.04 LTS
ROS 2 Jazzy
Python 3
OpenCV
NumPy
TurtleBot 4 ROS 2 interfaces
```

Environment setup:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=9
```

---

## ROS Topics

## Camera and perception

| Topic                                     | Type                           | Purpose                                 |
| ----------------------------------------- | ------------------------------ | --------------------------------------- |
| `/robot_09/oakd/rgb/image_raw/compressed` | `sensor_msgs/CompressedImage`  | OAK-D RGB camera input                  |
| `/robot_09/board_pose`                    | `geometry_msgs/PoseStamped`    | Raw board pose from ArUco detection     |
| `/robot_09/board_visible`                 | `std_msgs/Bool`                | Whether the board is currently detected |
| `/robot_09/board_used_ids`                | `std_msgs/Int32MultiArray`     | Marker IDs used for pose estimation     |
| `/robot_09/board_rpy`                     | `geometry_msgs/Vector3Stamped` | Roll/pitch/yaw diagnostics              |

## Tracking and prediction

| Topic                            | Type                        | Purpose                            |
| -------------------------------- | --------------------------- | ---------------------------------- |
| `/robot_09/tracked_board_pose`   | `geometry_msgs/PoseStamped` | Filtered or predicted board pose   |
| `/robot_09/tracker_status`       | `std_msgs/String`           | `measured`, `predicted`, or `lost` |
| `/robot_09/predicted_board_path` | `nav_msgs/Path`             | Short predicted target rollout     |

## Safety and control

| Topic               | Type                         | Purpose                             |
| ------------------- | ---------------------------- | ----------------------------------- |
| `/robot_09/scan`    | `sensor_msgs/LaserScan`      | LiDAR scan for front obstacle guard |
| `/robot_09/cmd_vel` | `geometry_msgs/TwistStamped` | Final velocity command to robot     |

---

## Installation

Clone into a ROS 2 workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

git clone https://github.com/Mobile-Robots-UGV/    turtlebot4-sft-aruco-kf-pf-recovery.git
```

Build:

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source ~/ros2_ws/install/setup.bash
```

For rebuilding only this project:

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select board_pose_ros sft_hardware_tracker
source ~/ros2_ws/install/setup.bash
```

---

## Running on TurtleBot 4 Hardware

Use `ROS_DOMAIN_ID=9` for robot 09.

Terminal 1: board pose estimation

```bash
export ROS_DOMAIN_ID=9
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch board_pose_ros board_pose.launch.py
```

Terminal 2: tracker and recovery follower

```bash
export ROS_DOMAIN_ID=9
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch sft_hardware_tracker sft_hardware_recovery.launch.py
```

---

## Expected Behavior

When the board is visible:

```text
tracker_status = measured
```

The robot follows the board using live filtered pose estimates.

When the board disappears briefly:

```text
tracker_status = predicted
```

The tracker predicts the target pose using the selected KF or PF backend. The follower moves cautiously and uses LiDAR front-range safety to avoid blindly driving into obstacles.

When the board remains missing too long:

```text
tracker_status = lost
```

The robot stops.

---

## Selecting KF or PF Tracking

The tracker backend is selected in:

```text
sft_hardware_tracker/config/sft_hardware_recovery.yaml
```

Use Kalman Filter:

```yaml
tracker_backend: kf
```

Use Particle Filter:

```yaml
tracker_backend: pf
```

Particle filter particle count:

```yaml
pf_num_particles: 300
```

---

## Important Configuration Parameters

File:

```text
sft_hardware_tracker/config/sft_hardware_recovery.yaml
```

### Tracker parameters

```yaml
tracker_backend: kf
pf_num_particles: 300
process_noise: 0.5
measurement_noise: 0.1
fresh_threshold_s: 0.5
prediction_timeout_s: 3.0
prediction_horizon_s: 1.5
prediction_dt_s: 0.1
publish_rate_hz: 20.0
```

### Follower parameters

```yaml
desired_distance_m: 0.70
min_distance_m: 0.30

kp_linear: 0.35
kp_angular: 0.90

max_linear_measured: 0.15
max_angular_measured: 0.45

max_linear_predicted: 0.02
max_angular_predicted: 0.12

pose_timeout_s: 3.0
publish_rate_hz: 20.0

scan_topic: /robot_09/scan
front_stop_distance_m: 0.45
front_slow_distance_m: 0.80
```

---

## Control Logic

The follower uses the tracked board pose:

```text
x = lateral offset of board in camera frame
z = forward distance to board in camera frame
```

The control law is:

```text
distance_error = z - desired_distance_m
linear.x = kp_linear * distance_error
angular.z = -kp_angular * x
```

The commands are clamped by safety limits.

If the board is too close, the robot is allowed to back up but not move forward.

If LiDAR detects an obstacle in front, forward velocity is slowed or stopped.

---

## KF/PF Tracking Logic

The tracker estimates the state of the board in camera-frame coordinates.

### Kalman Filter backend

State:

```text
[x, z, vx, vz]
```

where:

```text
x  = lateral board position
z  = forward board distance
vx = lateral velocity estimate
vz = forward velocity estimate
```

The KF smooths noisy ArUco detections and predicts the board pose when measurements are temporarily unavailable.

### Particle Filter backend

Particle state:

```text
[x, z, vx, vz]
```

The PF maintains multiple possible target states, updates particle weights from board pose measurements, resamples particles when needed, and predicts the target pose from the weighted state estimate.

---

## Debugging Commands

Check camera stream:

```bash
ros2 topic hz /robot_09/oakd/rgb/image_raw/compressed
```

Check raw board detection:

```bash
ros2 topic echo /robot_09/board_visible
ros2 topic echo /robot_09/board_pose
```

Check tracker status:

```bash
ros2 topic echo /robot_09/tracker_status
```

Check tracked pose:

```bash
ros2 topic echo /robot_09/tracked_board_pose
```

Check predicted path:

```bash
ros2 topic echo /robot_09/predicted_board_path
```

Check LiDAR:

```bash
ros2 topic hz /robot_09/scan
ros2 topic echo --once /robot_09/scan
```

Check robot command output:

```bash
ros2 topic echo /robot_09/cmd_vel geometry_msgs/msg/TwistStamped
```

Emergency stop command:

```bash
ros2 topic pub --once /robot_09/cmd_vel geometry_msgs/msg/TwistStamped "{header: {frame_id: base_link}, twist: {linear: {x: 0.0}, angular: {z: 0.0}}}"
```

---

## Testing Procedure

1. Launch `board_pose_ros`.
2. Place the ArUco board in front of the robot.
3. Confirm `/robot_09/board_visible` becomes `true`.
4. Confirm `/robot_09/board_pose` reports reasonable `x` and `z`.
5. Launch `sft_hardware_tracker`.
6. Confirm `/robot_09/tracker_status` becomes `measured`.
7. Move the board slowly and observe `/robot_09/tracked_board_pose`.
8. Hide the board briefly and observe status change to `predicted`.
9. Keep the board hidden and observe status change to `lost`.
10. Monitor `/robot_09/cmd_vel` to verify robot command behavior.

---

## Relation to Milestone 2

Milestone 2 established a hardware perception-to-control pipeline:

```text
camera calibration
ArUco board detection
6-DoF pose estimation
board following
safe stop on target loss
```

This repository builds on that work and extends it with:

```text
tracking state machine
KF/PF prediction
predicted target rollout
short target-loss recovery
LiDAR front-obstacle safety
hardware-ready TurtleBot 4 command output
```

---

## Relation to Milestone 3

For Milestone 3, the system moves from simple board following toward recovery-aware tracking.

This repository demonstrates:

* real-time ArUco board pose estimation on TurtleBot 4,
* selectable KF/PF tracking,
* `measured`, `predicted`, and `lost` target states,
* predicted target rollout,
* cautious recovery behavior during temporary target loss,
* LiDAR-based front obstacle protection,
* real robot velocity command generation.

---

## Current Limitations

This repository does not yet implement full global path planning around obstacles.

The current predicted path is a target-state prediction, not an obstacle-aware path. LiDAR is used as a local safety guard to slow or stop forward motion near obstacles.

The next step is to connect the predicted target goal to a map or costmap-based planner so that recovery behavior can navigate around walls and obstacles instead of only stopping in front of them.

---

## Future Work

Planned next steps:

* convert predicted target pose into a recovery goal,
* integrate map/costmap-based path planning,
* add obstacle-aware path following,
* add RViz visualization for predicted path and recovery state,
* add rosbag recording for measured/predicted/lost trials,
* compare KF and PF performance experimentally,
* add quantitative tracking metrics.

---

## Acknowledgments

This work builds on the team’s previous TurtleBot 4 OAK-D ArUco board pose and following implementation.

It also adapts the KF/PF tracker architecture idea from the simulation-side ArUco tracking and recovery repository into a hardware-focused TurtleBot 4 implementation.

---

## Authors

* Tatwik Meesala
* Prajjwal
* Lu Yan Tan

````

