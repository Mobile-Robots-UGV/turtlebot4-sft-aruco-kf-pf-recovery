# Smart Follower & Tracker TurtleBot 4 ROS 2

ROS 2 Jazzy hardware implementation of the **Smart Follower & Tracker** project on TurtleBot 4.

This repository implements a real-robot ArUco board following and tracking pipeline using the TurtleBot 4 OAK-D camera, calibrated board pose estimation, selectable KF/PF tracking, short-horizon prediction recovery, LiDAR-based safety checking, closed-loop velocity control, and SLAM-based mapping.

The system is designed for the RAS598 Mobile Robotics project milestone sequence and extends the earlier board-following work toward a more complete target tracking and recovery stack.

---

## Project Context

This repository supports the Smart Follower & Tracker project.

The larger project goal is to build a TurtleBot 4 system that can:

- detect an object of interest,
- estimate its relative pose,
- track it over time,
- follow it safely,
- handle temporary target loss,
- build a map of the environment while following,
- and prepare for future obstacle-aware recovery and planning.

This repository focuses on the hardware tracking, following, and mapping layer.

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
├── sft_hardware_tracker/
│   ├── sft_hardware_tracker/
│   │   ├── board_tracker_node.py
│   │   └── recovery_follower_node.py
│   ├── config/
│   │   └── sft_hardware_recovery.yaml
│   ├── launch/
│   │   └── sft_hardware_recovery.launch.py   ← master launch file
│   ├── package.xml
│   └── setup.py
├── RAS598 Mobile Robotics Final.rviz
└── README.md
```

---

## Package Overview

## 1. `board_pose_ros`

This package performs camera-based ArUco board pose estimation.

It subscribes to the TurtleBot 4 OAK-D compressed RGB image stream, detects the configured ArUco board, estimates its 6-DoF pose using camera calibration and known board geometry, and publishes ROS topics for downstream tracking and control.

### Updates to `board_pose_node.py`

The following improvements were made to the original node:

**1. Static TF broadcaster** — publishes a permanent `map → oak_camera_frame` transform so that the Fixed Frame in RViz2 is always valid, even when no board is detected. Previously, the Fixed Frame error blocked all RViz2 rendering.

**2. Debug image publisher** — publishes `/robot_09/board_debug_image` as a raw `sensor_msgs/Image` on every frame regardless of board detection. This eliminates the 20-second video latency that occurred when RViz2 subscribed directly to the uncompressed OAK-D stream over WiFi. The node now:
- receives the compressed image over WiFi (small bandwidth)
- decodes it locally on the VM using `cv2.imdecode`
- publishes the decoded frame with ArUco overlays locally to RViz2 (no WiFi hop)

**3. RViz2 MarkerArray** — publishes `/robot_09/board_markers` with a green sphere at the board center, a white line from the camera origin to the board, and text labels showing X/Y/Z distance and detected marker IDs.

**4. Subscriber queue depth = 1 with Best Effort QoS** — the image subscriber now uses `depth=1` and `BEST_EFFORT` reliability. This ensures only the latest camera frame is processed, preventing buffer buildup that caused control delay and oscillation.

**5. OpenCV debug overlay** — when a board is detected, the debug image shows:
- ArUco marker outlines drawn on frame
- 3D axis frame at board center
- Info panel with X, Y, Z, and Yaw values
- "No board detected" message when nothing is found

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
* broadcast a TF transform from camera frame to board frame,
* publish static TF `map → oak_camera_frame`,
* publish debug image with ArUco overlay,
* publish RViz2 MarkerArray for visualization.

### Main outputs

```text
/robot_09/board_pose
/robot_09/board_visible
/robot_09/board_used_ids
/robot_09/board_rpy
/robot_09/board_debug_image
/robot_09/board_markers
/tf  (oak_camera_frame → board_frame, dynamic)
/tf  (map → oak_camera_frame, static)
```

---

## 2. `sft_hardware_tracker`

This package adds the tracking, prediction, recovery, following, and mapping layer.

It contains two main nodes and one master launch file.

```text
board_tracker_node.py
recovery_follower_node.py
launch/sft_hardware_recovery.launch.py   ← launches everything
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
measured   = board is currently visible, using live filtered pose
predicted  = board recently lost, predicting pose using KF/PF
lost       = board missing too long, stop trusting prediction
```

### Tracker modes

| Status | Meaning | Behavior |
| ------ | ------- | -------- |
| `measured` | Board is currently visible | Use filtered live board pose |
| `predicted` | Board was recently lost | Predict target pose using KF/PF |
| `lost` | Board has been missing too long | Stop trusting prediction |

---

### `recovery_follower_node.py`

The follower subscribes to the tracked board pose, tracker status, and TurtleBot 4 LiDAR scan and publishes velocity commands.

### Updates to `recovery_follower_node.py`

The following improvements were made to reduce oscillation and improve following smoothness:

**1. Reduced angular gain** — `kp_angular` reduced from `0.90` to `0.45`. The original gain caused the robot to turn too aggressively, overshooting the center and oscillating.

**2. Added D term (PD controller)** — a derivative term `kd_angular = 0.08` was added to the angular control law. The D term resists fast changes in lateral error, dampening overshoot:

```
angular = -(kp_angular * x_error + kd_angular * x_dot)
```

where `x_dot` is the rate of change of the lateral error between control steps.

**3. Added deadband on X and Z errors** — small errors within a threshold are zeroed before computing velocity commands:
- `x_deadband_m = 0.08m` — robot does not steer if board is within 8cm of center
- `z_deadband_m = 0.05m` — robot does not drive if distance error is within 5cm

This prevents continuous micro-corrections when the board is approximately centered.

**4. Reduced max angular speed** — `max_angular_measured` reduced from `0.45` to `0.25 rad/s`. This limits how fast the robot can turn even when the error is large.

**5. Subscriber queue depth = 1 with Best Effort QoS** — all subscribers now use `depth=1` to always process only the latest pose, status, and scan data. This eliminates stale data buildup that caused delayed reactions and oscillation.

### Follower behavior

| Tracker Status | Follower Behavior |
| -------------- | ----------------- |
| `measured` | Normal visual following with PD control and deadband |
| `predicted` | Slow cautious predicted tracking with LiDAR safety |
| `lost` | Stop robot |

---

## System Architecture

```text
TurtleBot 4 OAK-D Camera
        |
        v (compressed over WiFi)
board_pose_node (decodes locally on VM)
        |
        | /robot_09/board_pose
        | /robot_09/board_visible
        | /robot_09/board_debug_image  → RViz2
        | /robot_09/board_markers      → RViz2
        v
board_tracker_node (KF or PF backend)
        |
        | /robot_09/tracked_board_pose
        | /robot_09/tracker_status
        | /robot_09/predicted_board_path
        v
recovery_follower_node (PD control + deadband)
        |
        | /robot_09/cmd_vel
        v
TurtleBot 4 Base

LiDAR safety:
/robot_09/scan → recovery_follower_node

SLAM (optional):
slam_toolbox (namespace: /robot_09)
        |
        | /map
        | /tf  (map → odom → base_link)
        v
RViz2 map visualization
```

---

## Hardware Platform

```text
Robot:   TurtleBot 4
Camera:  OAK-D
Sensor:  TurtleBot 4 LiDAR (rplidar)
Board:   Printed ArUco marker board (DICT_6X6_250, 4 markers)
```

Robot namespace:

```text
/robot_09
```

---

## Software Stack

```text
Ubuntu 24.04 LTS
ROS 2 Jazzy
Python 3
OpenCV (opencv-contrib-python >= 4.x)
NumPy
cv_bridge
tf2_ros
TurtleBot 4 ROS 2 interfaces
turtlebot4_navigation (for SLAM)
```

Environment setup:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=9
```

---

## ROS Topics

### Camera and perception

| Topic | Type | Purpose |
| ----- | ---- | ------- |
| `/robot_09/oakd/rgb/image_raw/compressed` | `sensor_msgs/CompressedImage` | OAK-D RGB camera input |
| `/robot_09/board_pose` | `geometry_msgs/PoseStamped` | Raw board pose from ArUco detection |
| `/robot_09/board_visible` | `std_msgs/Bool` | Whether the board is currently detected |
| `/robot_09/board_used_ids` | `std_msgs/Int32MultiArray` | Marker IDs used for pose estimation |
| `/robot_09/board_rpy` | `geometry_msgs/Vector3Stamped` | Roll/pitch/yaw diagnostics |
| `/robot_09/board_debug_image` | `sensor_msgs/Image` | Annotated debug image for RViz2 |
| `/robot_09/board_markers` | `visualization_msgs/MarkerArray` | Board position markers for RViz2 |

### Tracking and prediction

| Topic | Type | Purpose |
| ----- | ---- | ------- |
| `/robot_09/tracked_board_pose` | `geometry_msgs/PoseStamped` | Filtered or predicted board pose |
| `/robot_09/tracker_status` | `std_msgs/String` | `measured`, `predicted`, or `lost` |
| `/robot_09/predicted_board_path` | `nav_msgs/Path` | Short predicted target rollout |

### Safety and control

| Topic | Type | Purpose |
| ----- | ---- | ------- |
| `/robot_09/scan` | `sensor_msgs/LaserScan` | LiDAR scan for front obstacle guard |
| `/robot_09/cmd_vel` | `geometry_msgs/TwistStamped` | Final velocity command to robot |

### SLAM and localization

| Topic | Type | Purpose |
| ----- | ---- | ------- |
| `/map` | `nav_msgs/OccupancyGrid` | Built map from SLAM |
| `/robot_09/pose` | `geometry_msgs/PoseWithCovarianceStamped` | Robot position in map |

---

## Installation

Clone into a ROS 2 workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Mobile-Robots-UGV/turtlebot4-sft-aruco-kf-pf-recovery.git
```

Install dependencies:

```bash
pip3 install opencv-contrib-python --break-system-packages
sudo apt install ros-jazzy-cv-bridge ros-jazzy-tf2-geometry-msgs ros-jazzy-turtlebot4-navigation
```

If you have other packages with duplicate names in your workspace, ignore them:

```bash
touch ~/ros2_ws/src/<other-package-folder>/COLCON_IGNORE
```

Build:

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select board_pose_ros sft_hardware_tracker
source ~/ros2_ws/install/setup.bash
```

---

## Running on TurtleBot 4 Hardware

Run `set-ros-env robot` once before opening any terminals. Then use `export ROS_DOMAIN_ID=9` in each terminal.

### Quick Start — One Command

Launch everything at once (board_pose + tracker + follower + SLAM + RViz2):

```bash
export ROS_DOMAIN_ID=9
source ~/ros2_ws/install/setup.bash
ros2 launch sft_hardware_tracker sft_hardware_recovery.launch.py
```

### Optional arguments

```bash
# Without SLAM
ros2 launch sft_hardware_tracker sft_hardware_recovery.launch.py slam:=false

# Without RViz2
ros2 launch sft_hardware_tracker sft_hardware_recovery.launch.py rviz:=false

# Without SLAM and RViz2
ros2 launch sft_hardware_tracker sft_hardware_recovery.launch.py slam:=false rviz:=false
```

### Manual Launch (separate terminals)

If you prefer to launch each component separately:

**Terminal 1 — board pose estimation:**
```bash
export ROS_DOMAIN_ID=9
source ~/ros2_ws/install/setup.bash
ros2 launch board_pose_ros board_pose.launch.py
```

**Terminal 2 — tracker and recovery follower:**
```bash
export ROS_DOMAIN_ID=9
source ~/ros2_ws/install/setup.bash
ros2 launch sft_hardware_tracker sft_hardware_recovery.launch.py slam:=false rviz:=false
```

**Terminal 3 — SLAM:**
```bash
export ROS_DOMAIN_ID=9
ros2 launch turtlebot4_navigation slam.launch.py sync:=false namespace:=/robot_09
```

**Terminal 4 — RViz2:**
```bash
export ROS_DOMAIN_ID=9
rviz2 -d ~/ros2_ws/src/turtlebot4-sft-aruco-kf-pf-recovery-main/"RAS598 Mobile Robotics Final.rviz"
```

---

## RViz2 Setup

The saved RViz2 config `RAS598 Mobile Robotics Final.rviz` includes:

| Display | Topic | Purpose |
| ------- | ----- | ------- |
| Image | `/robot_09/board_debug_image` | Live camera with ArUco overlay |
| MarkerArray | `/robot_09/board_markers` | Board position in 3D |
| Map | `/map` | SLAM map |
| LaserScan | `/robot_09/scan` | LiDAR scan |
| RobotModel | — | Robot shape on map |
| Pose | `/robot_09/pose` | Robot position arrow |

Fixed Frame:
- `oak_camera_frame` — for camera-only view (no SLAM)
- `map` — for full SLAM map view

---

## SLAM Mapping

When SLAM is running, walk around with the ArUco board. The robot will follow you while building a map of the environment simultaneously.

Save the map when done:

```bash
ros2 run nav2_map_server map_saver_cli -f ~/my_map
```

This saves:
- `~/my_map.pgm` — map image
- `~/my_map.yaml` — map metadata

Check robot position in map:

```bash
ros2 topic echo /robot_09/pose
ros2 run tf2_ros tf2_echo map base_link
```

> **Key insight:** The `namespace:=/robot_09` argument tells SLAM to look for all topics under `/robot_09/` automatically. No topic remapping, bridge nodes, or yaml modifications are needed.

---

## Configuration

### Tuning the follower (`sft_hardware_recovery.yaml`)

Edit the yaml and relaunch — no rebuild needed:

```yaml
recovery_follower_node:
  ros__parameters:
    desired_distance_m: 0.70    # target follow distance
    kp_linear: 0.35             # forward/backward gain
    kp_angular: 0.45            # steering gain (lower = smoother)
    kd_angular: 0.08            # derivative damping (higher = less overshoot)
    x_deadband_m: 0.08          # ignore lateral errors smaller than this
    z_deadband_m: 0.05          # ignore distance errors smaller than this
    max_linear_measured: 0.15   # max forward speed (m/s)
    max_angular_measured: 0.25  # max turn speed (rad/s)
    front_stop_distance_m: 0.45 # LiDAR emergency stop distance
    front_slow_distance_m: 0.80 # LiDAR slow-down distance
```

### Tuning tips

| Problem | Fix |
| ------- | --- |
| Robot overshoots when turning | Reduce `kp_angular`, increase `kd_angular` |
| Robot jitters when centered | Increase `x_deadband_m` |
| Robot stops/starts too much | Increase `z_deadband_m` |
| Robot turns too slowly | Increase `kp_angular` |
| Oscillation persists | Reduce `max_angular_measured` |
| Robot does not rotate at all | Check `x_deadband_m` is not too large (e.g. 0.8 vs 0.08) |

### Selecting KF or PF Tracking

```yaml
tracker_backend: kf   # Kalman Filter (default, smooth)
tracker_backend: pf   # Particle Filter (better for sudden changes)
pf_num_particles: 300
```

---

## Expected Behavior

| Tracker Status | Robot Behavior |
| -------------- | -------------- |
| `measured` | Normal visual following with PD control and deadband |
| `predicted` | Very slow cautious movement, stops if obstacle detected |
| `lost` | Full stop |

---

## Debugging Commands

```bash
# Check camera stream
ros2 topic hz /robot_09/oakd/rgb/image_raw/compressed

# Check board detection
ros2 topic echo /robot_09/board_visible
ros2 topic echo /robot_09/board_pose

# Check tracker
ros2 topic echo /robot_09/tracker_status
ros2 topic echo /robot_09/tracked_board_pose

# Check LiDAR
ros2 topic hz /robot_09/scan

# Check velocity commands
ros2 topic echo /robot_09/cmd_vel geometry_msgs/msg/TwistStamped

# Check robot position in map (requires SLAM)
ros2 topic echo /robot_09/pose
ros2 run tf2_ros tf2_echo map base_link

# Emergency stop
ros2 topic pub --once /robot_09/cmd_vel geometry_msgs/msg/TwistStamped \
  "{header: {frame_id: base_link}, twist: {linear: {x: 0.0}, angular: {z: 0.0}}}"
```

---

## Testing Procedure

1. Launch everything: `ros2 launch sft_hardware_tracker sft_hardware_recovery.launch.py`
2. Place the ArUco board in front of the robot.
3. Confirm `/robot_09/board_visible` becomes `true`.
4. Confirm `/robot_09/board_debug_image` shows ArUco overlay in RViz2.
5. Confirm `/robot_09/tracker_status` becomes `measured`.
6. Move the board slowly — robot should follow smoothly.
7. Hide the board briefly — status should change to `predicted`.
8. Keep board hidden — status should change to `lost`, robot stops.
9. Check `/map` is building in RViz2 as you walk around.
10. Save map when done: `ros2 run nav2_map_server map_saver_cli -f ~/my_map`

---

## Relation to Milestone 2

Milestone 2 established the hardware perception-to-control pipeline:

```text
camera calibration → ArUco detection → 6-DoF pose estimation → board following → safe stop on target loss
```

---

## Relation to Milestone 3

This repository builds on Milestone 2 and extends it with:

```text
KF/PF tracking backend (measured / predicted / lost states)
predicted target rollout visualization
short target-loss recovery behavior
LiDAR front-obstacle safety guard
PD angular control with deadband for smooth following
low-latency debug image pipeline (compressed WiFi → local decode)
SLAM integration with namespace:=/robot_09
real-time map building while following
robot position in map frame
single-command master launch file
```

---

## Current Limitations

- No full global path planning around obstacles yet
- Predicted path is target-state prediction, not obstacle-aware
- LiDAR used as local safety guard only (stop/slow, not navigate around)
- Board pose visualization is in camera frame, not yet transformed to map frame

---

## Future Work

- Transform board pose into map frame for global visualization
- Connect predicted target goal to Nav2 for obstacle-aware recovery
- Add map/costmap-based path planning
- Add rosbag recording for measured/predicted/lost trials
- Compare KF and PF tracking performance experimentally
- Add quantitative tracking metrics

---

## Authors

- Tatwik Meesala
- Prajjwal
- Lu Yan Tan