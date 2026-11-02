import numpy as np
from scipy.spatial.transform import Rotation

from .data_classes import FossenState, FossenParameters, _vector



def fossen_acceleration(velocity,  theta, forces, restoring):
    """Return body nu_dot = solve(M, forces - C*nu - D*nu - restoring).
    """
    nu = _vector(velocity, 6, "velocity")
    theta = _vector(theta, 33, "theta")
    forces = _vector(forces, 6, "forces")
    restoring = _vector(restoring, 6, "restoring")
    mass = np.zeros((6, 6))
    upper = np.triu_indices(6)
    mass[upper] = theta[:21]
    mass[(upper[1], upper[0])] = theta[:21]
    try:
        np.linalg.cholesky(mass)
    except np.linalg.LinAlgError as error:
        raise ValueError("Total inertia must be positive definite for forward simulation.") from error
    momentum = mass @ nu
    coriolis = np.r_[np.cross(nu[3:], momentum[:3]),
                     np.cross(nu[:3], momentum[:3]) + np.cross(nu[3:], momentum[3:])]
    damping = theta[21:27] * nu + theta[27:] * np.abs(nu) * nu
    return np.linalg.solve(mass, forces - coriolis - damping - restoring)


def fossen_dynamics(x: FossenState, theta, forces, *, mass_kg, volume,
                    center_of_gravity=(0, 0, 0), center_of_buoyancy=(0, 0, 0),
                    water_density=1031.0, gravity=9.81):
    """Return x_dot for a fully submerged vehicle in still water (NED/body).

    x is a FossenState or a 13-vector ordered as position, quaternion, velocity.
    Returns a flat 13-vector of derivatives for numerical integration.
    Quaternion rotates BODY to NED; forces = [X,Y,Z,K,M,N] at the body origin.
    theta is the 33-vector from build_regression_fossen. Known mass_kg, volume
    (m³), and body-frame CG/CB offsets (m) determine restoring, not theta.
    Bind these constants with functools.partial to get f(x, theta, forces)
    for trace_dynamics. Supply applied wrench; actuator lag is not modelled.
    """
    if not isinstance(x, FossenState):
        x = FossenState.from_array(_vector(x, 13, "x"))
    cg = _vector(center_of_gravity, 3, "center_of_gravity")
    cb = _vector(center_of_buoyancy, 3, "center_of_buoyancy")
    constants = _vector([mass_kg, volume, water_density, gravity], 4, "physical constants")
    mass_kg, volume, water_density, gravity = constants
    if mass_kg <= 0 or volume < 0 or water_density <= 0 or gravity <= 0:
        raise ValueError("Mass, water density and gravity must be positive; volume nonnegative.")
    
    norm = np.linalg.norm(x.orientation)
    if norm < 1e-12:
        raise ValueError("Orientation quaternion must be nonzero.")
    quaternion = x.orientation / norm
    rotation = Rotation.from_quat(quaternion)
    
    down_body = rotation.inv().apply([0, 0, 1])
    weight = mass_kg * gravity * down_body
    buoyancy = -water_density * volume * gravity * down_body
    restoring = -np.r_[weight + buoyancy, np.cross(cg, weight) + np.cross(cb, buoyancy)]
    acceleration = fossen_acceleration(x.velocity, theta, forces, restoring=restoring)
    # q_dot = 1/2 q ⊗ [omega_body, 0], with scalar-last quaternions.
    q_dot = 0.5 * np.r_[quaternion[3] * x.velocity[3:] + np.cross(quaternion[:3], x.velocity[3:]),
                        -np.dot(quaternion[:3], x.velocity[3:])]
    return np.r_[rotation.apply(x.velocity[:3]), q_dot, acceleration]
