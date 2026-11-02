# Sysid: system-identification drone

`sysid` uses the Nautilus hull and thruster geometry scaled to 60%.
It keeps eight thrusters, odometry and an IMU for system identification.
Cameras, sonar, DVL, pressure sensor and the gripper (including its joints and
servos) are removed. Sensor rates, noise, drag settings and actuator response
match Nautilus. Odometry supplies position, orientation and body velocity;
the IMU supplies acceleration and angular velocity.

The base hull is approximately 0.54 × 0.435 × 0.30 m. Its mass is 11.5992 kg
(original mass × 0.6³), displaced volume is 0.0108320817855168 m³ and CG is
[0, 0, 0.006] m. Rigid-body inertia is
[0.8600256, 1.7970336, 2.0404224] kg*m² (original inertia × 0.6⁵).
Like Nautilus, it is slightly negatively buoyant in 1031 kg/m³ water; it needs
approximately 4.23 N upward thrust to hold depth. The gripper links are removed;
the base-hull properties are unchanged.

Thrusters keep Nautilus's linear force-command mapping and 0.05 s response,
with ±30 N limits in both simulator and allocator (previously ±15 N). Allocator
positions match the scaled simulator geometry. Manual gains are 20 N
translation and 2/2/4 N*m roll/pitch/yaw. Combined commands can saturate the allocator.

The `deepwater` scenario has still water, no waves or seabed, and starts the
vehicle at 100 m depth. The physical surface remains at NED z=0; this is a
deep test area, not an infinite fluid. Remain submerged during identification.
Stonefish defines a flat ocean surface even with waves disabled; see its
[environment documentation](https://stonefish.readthedocs.io/en/latest/environment.html).

Odometry retains Nautilus's 5 Hz rate; the IMU runs at 20 Hz. DVL recording remains disabled in `config/sysid_recording.yaml`.

## Files

Relative to `~/ros2_ws/src`:

- `vortex-stonefish-sim/stonefish_sim/drones/sysid.scn`: complete vehicle.
- `vortex-stonefish-sim/stonefish_sim/scenarios/deepwater.scn`: still-water world.
- `vortex-stonefish-sim/stonefish_sim/config/deepwater_config.yaml`: starting pose.
- `vortex-auv/auv_setup/config/robots/sysid.yaml`: allocator, topics and joystick gains.
- `vortex-auv/auv_setup/description/sysid.urdf.xacro`: body and IMU frames.

The robot config disables sonar forwarding and gripper forwarding in the
simulator bridge. A valid gripper topic is retained for the joystick node,
which requires it even when no gripper is fitted. The common bridge's DVL
topics remain configured but receive no sensor data.

Use the default `sysid` namespace; simulator topic names are explicit.

## Build and launch

Run in your ROS environment, after stopping the previous simulator:

```bash
cd ~/ros2_ws
source install/setup.bash
colcon build --packages-select stonefish_sim stonefish_sim_interface auv_setup system_identification
source install/setup.bash
ros2 launch stonefish_sim vortex_sim_launch.py \
  drone:=sysid scenario:=deepwater rendering:=true keyboard_joy:=true
```

Keep `rendering:=true` to view the vehicle. The launcher selects a different
scenario when rendering is disabled; no dedicated test-drone no-GPU scenario
is supplied. In the keyboard controller, select manual
(Xbox) mode with `1`, check that the killswitch is off, then use `W` for surge.

If there is still no thrust, inspect the command chain while commanding motion:

```bash
ros2 topic echo /sysid/wrench_input
ros2 topic echo /sysid/thruster_forces
ros2 topic echo /sysid/stonefish/thrusters
ros2 topic echo /sysid/thruster_state
```

Run these in separate terminals, or stop each with Ctrl-C before the next.
The first topic is requested force/torque, the next two carry eight thruster
commands, and the last is simulator feedback. Nonzero commands arriving at
`stonefish/thrusters` establish that the ROS forwarding path is working;
check simulator feedback and odometry to confirm actual thrust and motion.

## Record and evaluate

The normal launch loads `config/config.yaml`, which defaults to `/sysid/...`
topics and writes to `system_identification/data/sysid/`. The explicit
`config/sysid_recording.yaml` configuration is also available. Rebuild and
source the workspace after changing configuration.

In another terminal:

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch system_identification system_identification.launch.py
```

Exercise all axes at several amplitudes, keeping the hull submerged. Stop the
recorder before reading or resetting its files. New recordings go into
`data/sysid/`. Existing `data/test_drone/` recordings
are retained and can still be evaluated with `--vehicle test_drone`. The rename
and deeper starting pose do not change the current vehicle's mass or thrust
limits; older box-drone recordings use different physics.

```bash
cd ~/ros2_ws/src/vortex-auv/control/system_identification
python test/test_sim_full_fossen.py --vehicle sysid
```

This selects the scaled Nautilus hydrostatic constants and reads
`system_identification/data/sysid/`. To clear that directory's recordings:

```python
from pathlib import Path
from system_identification.src.data_handeling import reset_ros_csv_files
reset_ros_csv_files(Path("system_identification/data/sysid").glob("*_test.csv"))
```

Stonefish adds hydrodynamic inertia, so the supplied rigid-body inertia is not
the expected total-inertia fit. The diagnostic fits requested wrench; actuator
dynamics and allocation can make delivered thrust different. A smaller vehicle
alone does not guarantee accurate physical parameter estimates.
