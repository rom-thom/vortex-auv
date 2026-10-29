# Small Stonefish test drone

`test_drone` is a 0.30 × 0.20 × 0.15 m box with mass 9.279 kg. It is neutrally
buoyant in the scenario's 1031 kg/m³ water. CG and CB coincide at the body
origin, so the fully submerged hydrostatic wrench is zero. It starts at 2 m
depth, with no current, waves or seabed in its dedicated scenario.

Eight thrusters span all six DOFs. Their positions/directions match the ROS
allocator, with ±3 N limits, a linear command-to-thrust mapping and a 0.02 s
first-order actuator response. Joystick gains are reduced to 4 N translation
and 0.2/0.2/0.4 N*m roll/pitch/yaw. Combined commands can still saturate.
Odometry and an IMU at the body origin publish at 50 Hz, without configured
measurement noise. There are no cameras, DVL or gripper.

## Files

Relative to `~/ros2_ws/src`:

- `vortex-stonefish-sim/stonefish_sim/drones/test_drone.scn`: physics/sensors.
- `vortex-stonefish-sim/stonefish_sim/scenarios/test_drone.scn`: still-water world.
- `vortex-auv/auv_setup/config/robots/test_drone.yaml`: allocator and ROS topics.
- `vortex-auv/auv_setup/description/test_drone.urdf.xacro`: body and IMU frames.

The scenario uses existing materials, looks and propeller mesh assets. Use the
default `test_drone` namespace because the simulator topic names are explicit.

## Build and launch

Run in the ROS environment where you normally launch the simulator:

```bash
cd ~/ros2_ws
source install/setup.bash
colcon build --packages-select stonefish_sim auv_setup system_identification
source install/setup.bash
ros2 launch stonefish_sim vortex_sim_launch.py \
  drone:=test_drone scenario:=test_drone rendering:=true keyboard_joy:=true
```

Keep `rendering:=true`: the current launcher otherwise substitutes a different
no-GPU scenario for unlisted scenario names. A no-GPU variant is not supplied.

In another terminal, record using the separate output directory:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run system_identification system_identification_node --ros-args \
  --params-file src/vortex-auv/control/system_identification/config/test_drone_recording.yaml
```

Exercise all axes at several amplitudes, keeping the hull submerged. Stop the
recorder before reading or resetting its files. Evaluate the recording with:

```bash
cd ~/ros2_ws/src/vortex-auv/control/system_identification
python test/test_sim_full_fossen.py --vehicle test_drone
```

This selects the new drone's hydrostatic constants and reads
`system_identification/data/test_drone/`. Nautilus recordings and defaults are
kept separate. To reset only the new drone's recordings from this directory:

```python
from pathlib import Path
from system_identification.src.data_handeling import reset_ros_csv_files
reset_ros_csv_files(Path("system_identification/data/test_drone").glob("*_test.csv"))
```

## What this experiment establishes

The specified rigid-body inertia diagonal is
`[0.048328125, 0.086990625, 0.1005225]` kg*m². Stonefish adds hydrodynamic inertia,
so these are not the expected total-inertia regression coefficients. Likewise,
the configured drag coefficients are geometric hydrodynamic settings, not
direct targets for the fitted damping coefficients. Neutral buoyancy and low
mass alone do not guarantee a physically valid fit.

The recorded-data test uses requested wrench; actuator dynamics and allocation
can cause differences from delivered thrust. Use the existing synthetic test
to verify exact parameter recovery, and this drone to test the ROS recording
and simulator identification pipeline.
