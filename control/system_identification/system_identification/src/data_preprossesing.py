import numpy as np




def remove_gravity(accelerations: np.ndarray, orientation_quaternion: np.ndarray, gravity_world: np.ndarray=np.array([0.0, 0.0, 9.81])):
    """accelerations: [acceleration_x, acceleration_y, acceleration_z]
    orientation_quaternion: [cos(theta/2), sin(theta/2) * r_1, sin(theta/2) * r_2, sin(theta/2) * r_3]"""
    # TODO continue