"""Append individual ROS messages to separate numeric CSV files."""

from collections.abc import Sequence
from numbers import Real

import numpy as np

from system_identification.src.data_io import write_to_file


def ros_message_to_data(message, timestamp=None) -> dict[str, np.ndarray]:
    """Convert one ROS 2 message into a dict with key = name, 1 dimentional np.arrays.

    time is header.stamp in seconds, unless timestamp is supplied. For
    headerless messages, supply the ROS receive time in seconds explicitly.
    Raw stamp.sec and stamp.nanosec fields are also kept as separate columns.
    No synchronization, interpolation, or sensor correction is performed.

    Each numeric scalar becomes a length-one NumPy array, for example
    {"time": np.array([1.5]), "linear_acceleration.x": np.array([0.2])}.
    """
    if not callable(getattr(message, "get_fields_and_field_types", None)):
        raise TypeError("Expected one ROS 2 message, not a list or scalar.")

    if timestamp is None:
        try:
            stamp = message.header.stamp
            timestamp = stamp.sec + stamp.nanosec * 1e-9
        except AttributeError as error:
            raise ValueError(
                "A message without header.stamp requires timestamp in seconds."
            ) from error
    if not isinstance(timestamp, Real) or not np.isfinite(timestamp):
        raise ValueError("timestamp must be a finite number of seconds.")

    data = {"time": np.asarray([timestamp], dtype=float)}

    def add_fields(value, name):
        if isinstance(value, (str, bytes)):
            return
        if callable(getattr(value, "get_fields_and_field_types", None)):
            for field in value.get_fields_and_field_types():
                path = f"{name}.{field}" if name else field
                # Reserve 'time' for the recording timestamp.
                if not name and field == "time":
                    path = "message.time"
                add_fields(getattr(value, field), path)
        elif isinstance(value, (Real, np.bool_)):
            data[name] = np.asarray([value], dtype=float)
        elif isinstance(value, (Sequence, np.ndarray)):
            for index, item in enumerate(value):
                add_fields(item, f"{name}[{index}]")
        else:
            raise TypeError(f"Unsupported ROS field {name!r}: {type(value).__name__}")

    add_fields(message, "")
    return data


def write_ros_data(message, file_path: str, timestamp=None, append: bool = True) -> dict[str, np.ndarray]:
    """Convert one ROS 2 message, write it to CSV, and return its data dict."""

    data = ros_message_to_data(message, timestamp=timestamp)
    write_to_file(data, file_path, append=append)
    return data
