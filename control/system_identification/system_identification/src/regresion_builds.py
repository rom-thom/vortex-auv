
import numpy as np


FOSSEN_PARAMETER_NAMES = (
    tuple(f"M_{i + 1}{j + 1}" for i in range(6) for j in range(i, 6))
    + tuple(f"d_linear_{axis}" for axis in ("u", "v", "w", "p", "q", "r"))
    + tuple(f"d_quadratic_{axis}" for axis in ("u", "v", "w", "p", "q", "r"))
)


def build_regression_fossen(data):
    """Build A @ theta = tau - g for the 6-DOF Fossen model.

    Required finite arrays, all (N, 6), N > 0:
      velocity: water-relative BODY velocity nu_r = [u, v, w, p, q, r].
      acceleration: time derivative of those components, d(nu_r)/dt.
      force: applied BODY wrench tau = [X, Y, Z, K, M, N], in N and N*m.
      restoring: known left-hand-side hydrostatic wrench g(eta); supply
          explicit zeros only when restoring forces/moments are negligible.

    Fits M*nu_r_dot + C(M, nu_r)*nu_r + Dl*nu_r + Dq*(abs(nu_r)*nu_r).
    M is constant symmetric total inertia (rigid body + added mass); Dl/Dq
    are diagonal. Parameter order is FOSSEN_PARAMETER_NAMES: 21 upper-triangle
    entries of M, six linear and six quadratic positive-damping coefficients.
    The same inertia parameters determine C, preserving its skew symmetry.

    Returns A (6*N, 33), b (6*N,), stacked sample first, then DOF.
    Uses SI units and one common body origin/frame. Assumes no current or a
    spatially uniform, steady, irrotational current; no unmodelled external
    loads. Corrected IMU acceleration is NOT nu_r_dot during rotation.
    See docs/fossen_regression.md for preparation, usage and limitations.
    """
    arrays = {}
    for name in ("velocity", "acceleration", "force", "restoring"):
        if name not in data:
            raise ValueError(f"Missing required Fossen input: {name!r}.")
        try:
            if np.iscomplexobj(data[name]):
                raise ValueError("Complex values are not supported.")
            values = np.asarray(data[name], dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must contain real numeric values.") from error
        if values.ndim != 2 or values.shape[1] != 6 or values.shape[0] == 0:
            raise ValueError(f"{name} must have nonempty shape (N, 6).")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
        arrays[name] = values

    velocity = arrays["velocity"]
    if any(values.shape != velocity.shape for values in arrays.values()):
        raise ValueError("All Fossen inputs must have matching shape (N, 6).")
    acceleration = arrays["acceleration"]
    linear, angular = velocity[:, :3], velocity[:, 3:]
    regression = np.zeros((len(velocity), 6, len(FOSSEN_PARAMETER_NAMES)))

    # Each basis matrix contributes to BOTH M*nu_dot and C(M, nu)*nu.
    for column, (i, j) in enumerate(zip(*np.triu_indices(6))):
        basis = np.zeros((6, 6))
        basis[i, j] = basis[j, i] = 1.0
        momentum = velocity @ basis.T
        contribution = acceleration @ basis.T
        contribution[:, :3] += np.cross(angular, momentum[:, :3])
        contribution[:, 3:] += (
            np.cross(linear, momentum[:, :3])
            + np.cross(angular, momentum[:, 3:])
        )
        regression[:, :, column] = contribution

    axes = np.arange(6)
    regression[:, axes, 21 + axes] = velocity
    regression[:, axes, 27 + axes] = np.abs(velocity) * velocity
    target = arrays["force"] - arrays["restoring"]
    return regression.reshape(-1, len(FOSSEN_PARAMETER_NAMES)), target.reshape(-1)






def build_regression_1D(data):
    """Build F = d₁*v + d₂*v|v| + m*a to estimate [d₁, d₂, m].
    data = {
        "velocity":     np.array([0.1, 0.2]),     # m/s
        "acceleration": np.array([0.5, 0.4]),     # m/s²
        "force":        np.array([28.0, 35.0]),   # N
    }
    """

    velocity = np.asarray(data["velocity"], dtype=float).reshape(-1)
    acceleration = np.asarray(data["acceleration"], dtype=float).reshape(-1)
    force = np.asarray(data["force"], dtype=float).reshape(-1)


    if velocity.size == 0:
        raise ValueError("At least one sample is required.")
    if not (velocity.shape == acceleration.shape == force.shape):
        raise ValueError(
            "velocity, acceleration, and force must have matching lengths."
        )
    if not all(np.all(np.isfinite(values)) for values in (velocity, acceleration, force)):
        raise ValueError("All data values must be finite.")

    # d1*v + d2*v*|v| + mass * acceleration = force
    A = np.column_stack((velocity, velocity * np.abs(velocity), acceleration))
    b = force
    
    return A, b
