import numpy as np
from scipy.spatial.transform import Rotation

def remove_gravity(
    accelerations: np.ndarray,
    orientation_quaternion: np.ndarray,
    gravity_world=(0.0, 0.0, 9.81),
) -> np.ndarray:
    """Remove gravity from raw IMU specific force (m/s²)
    Inputs:
        acceleration: [[0.6, 0.5], [1.1, 1.0], [-10, -9.9]]
        orientation quaternion: [[x_1, y_1, z_1, w_1], [x_2, y_2, ...], ...]

    Input: accelerations (3,) or (N, 3); sensor-to-world quaternions [x,y,z,w]
    (4,) shared or matching (N, 4), normalized automatically.
    gravity_world: (3,) in m/s²; default z-down, use (0, 0, -9.81) for z-up.
    Returns a new array of the same shape in sensor axes; inputs are unchanged.
    Raises ValueError for invalid shapes, empty/nonfinite/nonreal data or zero quaternions.
    """
    arrays = []
    for name, values in (
        ("accelerations", accelerations),
        ("orientation_quaternion", orientation_quaternion),
        ("gravity_world", gravity_world)):

        try:
            if np.iscomplexobj(values):
                raise ValueError("Complex values are not supported.")
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must contain real numeric values.") from error
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
        arrays.append(array)

    accelerations, quaternion, gravity = arrays
    if not (
        accelerations.shape == (3,)
        or (accelerations.ndim == 2 and accelerations.shape[1] == 3
            and accelerations.shape[0] > 0)
    ):
        raise ValueError("accelerations must have shape (3,) or nonempty (N, 3).")
    if quaternion.shape != (4,) and not (
        accelerations.ndim == 2
        and quaternion.shape == (accelerations.shape[0], 4)
    ):
        raise ValueError("orientation_quaternion must have shape (4,) or matching (N, 4).")
    if gravity.shape != (3,):
        raise ValueError("gravity_world must have shape (3,).")

    # Scale before normalization to avoid overflow or underflow in the norm.
    scale = np.max(np.abs(quaternion), axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("orientation_quaternion must not contain zero quaternions.")
    quaternion = quaternion / scale
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)


    # This is the real deal
    rotation = Rotation.from_quat(quaternion)
    gravity_sensor = rotation.inv().apply(gravity)
    return accelerations + gravity_sensor



if __name__ == "__main__":
    from data_handeling import merge_file_data
    data = merge_file_data(["system_identification/data/imu_test.csv"])
    accelerations = np.column_stack([
        data["linear_acceleration.x"],
        data["linear_acceleration.y"],
        data["linear_acceleration.z"],
    ])

    quaternions = np.column_stack([
        data["orientation.x"],
        data["orientation.y"],
        data["orientation.z"],
        data["orientation.w"],
    ])
    print(remove_gravity(accelerations, quaternions)[0:20])