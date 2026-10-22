import numpy as np
import yaml

with open("../../auv_setup/config/robots/nautilus.yaml") as file: config = yaml.safe_load(file)

thrusters = config["/**"]["ros__parameters"]["propulsion"]["thrusters"]

directions = np.asarray(thrusters["thruster_force_direction"], dtype=float).reshape(3, thrusters["num"]).T


def motor_thrust_to_body_force_and_torque_body(motor_directions, thrust_array):
    force_xyz = np.array([0, 0, 0])
    torque_xyz = np.array([0, 0, 0])

    for dir, thrust in zip(motor_directions, thrust_array):
        force_xyz += thrust * dir


    return force_xyz, torque_xyz
