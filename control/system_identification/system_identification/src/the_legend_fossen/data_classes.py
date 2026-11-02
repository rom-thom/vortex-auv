import numpy as np
from scipy.spatial.transform import Rotation





from dataclasses import dataclass

@dataclass
class FossenState:
    position: np.ndarray      # [north, east, down]
    orientation: np.ndarray   # [qx, qy, qz, qw], body → NED
    velocity: np.ndarray      # [u, v, w, p, q, r], bodyframe

    def __post_init__(self):
        self.position = _vector(self.position, 3, "position").copy()
        self.orientation = _vector(self.orientation, 4, "orientation").copy()
        self.velocity = _vector(self.velocity, 6, "velocity").copy()

    def as_array(self):
        return np.concatenate([
            self.position,
            self.orientation,
            self.velocity,
        ])

    @classmethod
    def from_array(cls, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (13,):
            raise ValueError("FossenState krev 13 verdiar.")
        return cls(x[:3].copy(), x[3:7].copy(), x[7:].copy())


@dataclass
class FossenParameters:
    inertia: np.ndarray           # (6, 6), symmetric total inertia
    linear_damping: np.ndarray    # (6,)
    quadratic_damping: np.ndarray # (6,)

    def as_array(self):
        return np.concatenate([
            self.inertia[np.triu_indices(6)],
            self.linear_damping,
            self.quadratic_damping,
        ])

    @classmethod
    def from_array(cls, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (33,):
            raise ValueError("Expected 33 Fossen parameters.")

        inertia = np.zeros((6, 6))
        upper = np.triu_indices(6)
        inertia[upper] = theta[:21]
        inertia[(upper[1], upper[0])] = theta[:21]

        return cls(inertia, theta[21:27].copy(), theta[27:].copy())








def _vector(value, size, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real.")
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite vector of length {size}.")
    return value
