# ME5413 Final Project (ROS1 Workspace)

This workspace contains the simulation, mapping, localization, navigation, and perception stack for the ME5413 final project.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Workspace Structure](#workspace-structure)
- [Build](#build)
- [Run](#run)
- [Mapping](#mapping)
- [Navigation](#navigation)
- [Key Topics](#key-topics)
- [Troubleshooting](#troubleshooting)

## Overview
The project is built on ROS1 (`catkin`) and uses:
- Gazebo + Jackal simulation environment
- `slam_toolbox` for map building
- `map_server + AMCL + move_base` for navigation
- Phase-based task scheduling (`task_executor.py`)
- Vision-based perception for mission logic (`easyocr`, YOLOv5, room digit detector)

## Requirements
Recommended runtime:
- Ubuntu 20.04
- ROS Noetic
- Python 3

Core ROS packages (used by this workspace):
- `gazebo_ros`
- `map_server`
- `amcl`
- `move_base` (via `jackal_navigation`)
- `slam_toolbox`
- `rviz`
- `teleop_twist_keyboard`
- `jackal_description`, `jackal_control`, `jackal_navigation`, `jackal_gazebo`

Python dependencies:
- `opencv-python`
- `cv_bridge` (ROS)
- `easyocr`
- `torch` (for YOLOv5-based nodes)

Notes:
- `easyocr` and `torch` are required by perception nodes.
- If you use GPU acceleration, install CUDA-compatible Torch.

## Workspace Structure
- `me5413_world/`: project world, launch files, task scheduler
- `yolov5_detector/`: perception nodes (cone detector, floor digit OCR, room digit OCR)
- `slam_toolbox/`: SLAM package source
- `amcl/`: AMCL localization package source
- `jackal_description/`: Jackal model and URDF
- `interactive_tools/`: RViz panel plugin

## Build
From workspace root (parent of `src`):

```bash
catkin_make
source devel/setup.bash
```

## Run
Typical run order:

1. Start world:
```bash
roslaunch me5413_world world.launch
```

2. Then choose one mode:
- Mapping mode:
```bash
roslaunch me5413_world slam_toolbox_mapping.launch
```
- Navigation mode:
```bash
roslaunch me5413_world navigation.launch
```

## Mapping
Mapping uses `slam_toolbox` (online async mode) launched by:
- `me5413_world/launch/slam_toolbox_mapping.launch`

What it does:
- Starts teleop for manual driving
- Runs `slam_toolbox` (`online_async.launch`)
- Opens RViz config for SLAM visualization

Suggested workflow:
1. Launch `world.launch`
2. Launch `slam_toolbox_mapping.launch`
3. Drive robot to cover free space and corridors
4. Save map when coverage is sufficient
5. Use saved map in navigation (`map_server + AMCL`)

## Navigation
Navigation in this project includes scheduling, localization, planning/control, and perception-driven decisions.

### 1. Scheduling (Task Orchestration)
Main node:
- `me5413_world/scripts/task_executor.py`

Responsibilities:
- Publishes mission phase on `/task_phase`
- Drives multi-phase mission execution:
  - Phase 1 patrol
  - Phase 2 exit/ramp handling
  - Phase 3 corridor traversal
  - Phase 4 cone check (`/blockornot`)
  - Phase 5 room-by-room digit matching
- Uses `/leastcount` (from floor OCR) and `/detectnumber` (room OCR) for final decision

### 2. Localization
Localization stack:
- `map_server` (loads static map)
- `amcl` (particle filter localization)
- `initial_pose_publisher` (bootstraps AMCL initial pose)

Launch entry:
- `me5413_world/launch/navigation.launch`

### 3. Navigation (Planning and Execution)
Planner/controller:
- `move_base` (from `jackal_navigation`)

Executor behavior:
- Sends goals to `move_base`
- Uses AMCL pose distance threshold for arrival checks
- Applies per-goal timeout handling

### 4. Perception
Navigation mode starts:
- `block_detector_yolov5_node.py`
  - Publishes `/blockornot` during phase-based cone checking
- `easyocr_digit_node.py`
  - Publishes `/leastcount` for floor digit statistics
- `room_digit_detector_node.py`
  - Enabled by `/room_digit_detector_enable`
  - Publishes `/detectnumber` for each room in Phase 5

## Key Topics
- `/task_phase` (`std_msgs/Int32`): mission phase indicator
- `/waypoint/next_goal` (`geometry_msgs/PoseStamped`): current target waypoint
- `/amcl_pose` (`geometry_msgs/PoseWithCovarianceStamped`): localization output
- `/blockornot` (`std_msgs/Bool`): obstacle/cone decision
- `/leastcount` (`std_msgs/Int32`): least frequent digit from floor scan
- `/room_digit_detector_enable` (`std_msgs/Bool`): room OCR switch
- `/detectnumber` (`std_msgs/Int32`): room OCR result
- `/cmd_unblock` (`std_msgs/Bool`): trigger for temporary obstacle unblock logic

## Troubleshooting
- If OCR nodes fail to start, verify Python packages (`easyocr`, `torch`, OpenCV).
- If `move_base` is unstable, check:
  - TF tree (`map -> odom -> base_link`)
  - Laser frame (`tim551`) configuration in launch
  - AMCL convergence and map alignment
- If Phase 5 never matches:
  - Confirm `/leastcount` is published before Phase 5
  - Confirm `/detectnumber` is published in each room
  - Check camera visibility and OCR confidence threshold

