from functools import partial

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from system_identification.src.fossen_model import fossen_acceleration, fossen_dynamics
from system_identification.src.regresion_builds import build_regression_fossen
from system_identification.src.evaluate import trace_dynamics


def parameters(mass=None):
    if mass is None:
        mass = np.diag([10., 11., 12., 2., 3., 4.])
    return np.r_[mass[np.triu_indices(6)], np.ones(6), 2 * np.ones(6)]


def test_inverse_matches_regression_with_coupled_inertia():
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(6, 6))
    theta = parameters(matrix.T @ matrix + np.eye(6))
    for _ in range(10):
        nu, acceleration, g = rng.normal(size=(3, 6))
        A, _ = build_regression_fossen(dict(velocity=nu[None], acceleration=acceleration[None],
                                           force=np.zeros((1, 6)), restoring=g[None]))
        force = A @ theta + g
        np.testing.assert_allclose(fossen_acceleration(nu, theta, force, restoring=g), acceleration, atol=1e-12)


def test_static_restoring_and_rotation():
    x = np.zeros(13)
    x[3:7] = Rotation.from_euler('x', np.pi / 2).as_quat()
    # Weight 98.1 N, no buoyancy, CG 0.1 m down: oppose force +y and torque -x.
    force = [0, -98.1, 0, 9.81, 0, 0]
    dx = fossen_dynamics(x, parameters(), force, mass_kg=10, volume=0,
                         center_of_gravity=[0, 0, .1])
    np.testing.assert_allclose(dx, 0, atol=1e-12)
    x[7:10] = [0, 1, 0]
    dx = fossen_dynamics(x, parameters(), force, mass_kg=10, volume=0)
    np.testing.assert_allclose(dx[:3], [0, 0, 1], atol=1e-12)


def test_integration_constant_surge_and_yaw():
    theta = parameters(np.eye(6)); theta[21:] = 0
    f = partial(fossen_dynamics, mass_kg=1, volume=1/1031)
    x = np.zeros(13); x[6] = 1
    time = np.linspace(0, 2, 21)
    trajectory = trace_dynamics(x, f, time, theta, input_forces=[1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(trajectory[:, 0], .5*time**2, atol=1e-8)
    np.testing.assert_allclose(trajectory[:, 7], time, atol=1e-8)
    x[12] = 1
    trajectory = trace_dynamics(x, f, time, theta, input_forces=np.zeros(6), rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(trajectory[:, 5], np.sin(time/2), atol=1e-6)
    np.testing.assert_allclose(trajectory[:, 6], np.cos(time/2), atol=1e-6)


def test_invalid_mass_and_quaternion():
    theta = parameters(); theta[0] = -1
    with pytest.raises(ValueError, match='positive definite'):
        fossen_acceleration(np.zeros(6), theta, np.zeros(6), restoring=np.zeros(6))
    with pytest.raises(ValueError, match='quaternion'):
        fossen_dynamics(np.zeros(13), parameters(), np.zeros(6), mass_kg=1, volume=0)


def test_fossen_state_with_list_fields_and_integrator():
    from system_identification.src.fossen_model import FossenState

    start = FossenState([0, 0, 100], [0, 0, 0, 1], [0] * 6)
    theta = parameters(np.eye(6))
    theta[21:] = 0
    f = partial(fossen_dynamics, mass_kg=1, volume=1 / 1031)
    force = [1, 0, 0, 0, 0, 0]
    np.testing.assert_allclose(f(start, theta, force), f(start.as_array(), theta, force))

    def state_dynamics(state, theta, force):
        assert isinstance(state, FossenState)
        return f(state, theta, force)

    time = np.linspace(0, 1, 11)
    result = trace_dynamics(start, state_dynamics, time, theta,
                            input_forces=force, states=[0, 7])
    np.testing.assert_allclose(result[:, 0], .5 * time**2, atol=1e-8)
    np.testing.assert_allclose(result[:, 1], time, atol=1e-8)
    np.testing.assert_array_equal(start.velocity, np.zeros(6))
    with pytest.raises(ValueError, match='orientation'):
        FossenState([0, 0, 0], [0, 0, 1], [0] * 6)
