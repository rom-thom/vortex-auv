"""Fit the full Fossen model to recorded simulator data, assuming still water.

Run: python test/test_sim_full_fossen.py
Uses odom_test.csv and wrench_input_test.csv, not synthetic measurements.
The wrench is a command, so this is a model-fit diagnostic, not ground truth.
"""

import argparse
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from system_identification.src.data_handeling import load_data, merge_data
from system_identification.src.evaluate import evaluate, plot_predictions, print_evaluation
from system_identification.src.regresion_builds import (
    FOSSEN_PARAMETER_NAMES,
    build_regression_fossen,
)
from system_identification.src.system_identification import SystemIdentification


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "system_identification" / "data"

# Nautilus default.scn / nautilus.scn assumptions. Update if the model changes.
MASS_KG = 53.7
WATER_DENSITY = 1031.0
# Volume and centroid of nautilus_collision.obj (fully submerged base hull).
DISPLACED_VOLUME = 0.0501485267848
CENTER_OF_BUOYANCY = np.array([0.0, 0.0, 0.0])
CENTER_OF_GRAVITY = np.array([0.0, 0.0, 0.01])
MIN_DEPTH = 0.7  # Conservative hull clearance below the NED water surface.

VEHICLES = {
    "nautilus": (MASS_KG, DISPLACED_VOLUME, CENTER_OF_GRAVITY, CENTER_OF_BUOYANCY, MIN_DEPTH),
    "sysid": (
        MASS_KG * 0.6**3, DISPLACED_VOLUME * 0.6**3,
        CENTER_OF_GRAVITY * 0.6, CENTER_OF_BUOYANCY * 0.6, MIN_DEPTH * 0.6,
    ),
}

# Preserve evaluation of recordings made before the vehicle was renamed.
VEHICLES["test_drone"] = VEHICLES["sysid"]


def _prepare_segment(segment, vehicle="nautilus"):
    """Differentiate within this split only; exclude edges and surface samples."""
    if len(segment["time"]) < 4:
        raise ValueError("Each split needs at least four odometry samples.")
    acceleration = np.gradient(
        segment["velocity"], segment["time"] - segment["time"][0], axis=0,
    )
    rotation = Rotation.from_quat(segment["orientation"])
    down_body = rotation.inv().apply([0.0, 0.0, 1.0])
    mass_kg, volume, center_of_gravity, center_of_buoyancy, min_depth = VEHICLES[vehicle]
    weight = mass_kg * 9.81 * down_body
    buoyancy = -WATER_DENSITY * volume * 9.81 * down_body
    # Fossen g is the negative of the external weight/buoyancy wrench.
    restoring = -np.column_stack([
        weight + buoyancy,
        np.cross(center_of_gravity, weight) + np.cross(center_of_buoyancy, buoyancy),
    ])
    keep = segment["depth"] >= min_depth
    keep[[0, -1]] = False
    if np.count_nonzero(keep) < 2:
        raise ValueError(f"Too few submerged samples: record more motion below {min_depth} m.")
    return {
        "time": segment["time"][keep],
        "velocity": segment["velocity"][keep],
        "acceleration": acceleration[keep],
        "force": segment["force"][keep],
        "restoring": restoring[keep],
    }


def run_sim_fit(directory=None, vehicle="nautilus", *, plot=False, save_plot=None):
    if directory is None:
        directory = DATA_DIRECTORY if vehicle == "nautilus" else DATA_DIRECTORY / vehicle
    odom = load_data(Path(directory) / "odom_test.csv")
    wrench = load_data(Path(directory) / "wrench_input_test.csv")
    velocity_names = [f"twist.twist.{kind}.{axis}"
                      for kind in ("linear", "angular") for axis in "xyz"]
    orientation_names = [f"pose.pose.orientation.{axis}" for axis in "xyzw"]
    force_names = [f"wrench.{kind}.{axis}"
                   for kind in ("force", "torque") for axis in "xyz"]
    signals = {
        name: (odom["time"], odom[name])
        for name in velocity_names + orientation_names + ["pose.pose.position.z"]
    }
    signals.update({name: (wrench["time"], wrench[name]) for name in force_names})
    # Use native odometry timestamps, rather than upsampling to the wrench rate.
    merged = merge_data(signals, target_time=odom["time"])
    data = {
        "time": merged["time"],
        "velocity": np.column_stack([merged[name] for name in velocity_names]),
        "orientation": np.column_stack([merged[name] for name in orientation_names]),
        "force": np.column_stack([merged[name] for name in force_names]),
        "depth": merged["pose.pose.position.z"],
    }
    model = SystemIdentification(regression_builder=build_regression_fossen)
    training, validation = model.split_data(data, 0.7)
    training, validation = _prepare_segment(training, vehicle), _prepare_segment(validation, vehicle)
    A, _ = model.build_regression(training)
    rank = np.linalg.matrix_rank(A)
    norms = np.linalg.norm(A, axis=0)
    condition = np.linalg.cond(A / np.where(norms > 0, norms, 1.0))
    # Inertia entries may have either sign; all 12 damping coefficients must be nonnegative.
    lower = np.full(len(FOSSEN_PARAMETER_NAMES), -np.inf)
    upper = np.full(len(FOSSEN_PARAMETER_NAMES), np.inf)
    lower[21:] = 0.0
    parameters = model.fit_parameters(training, bounds=(lower, upper))
    predicted = model.predict(validation).reshape(-1, 6) + validation["restoring"]
    actual = validation["force"]
    output_names = list("XYZKMN")
    report = evaluate(actual, predicted, output_names)
    rmse = np.array([report[name]["rmse"] for name in output_names])
    baseline = np.broadcast_to(training["force"].mean(axis=0), actual.shape)
    baseline_report = evaluate(actual, baseline, output_names)
    for name in output_names:
        report[name]["baseline_rmse"] = baseline_report[name]["rmse"]

    mass = np.zeros((6, 6))
    upper = np.triu_indices(6)
    mass[upper] = parameters[:21]
    mass[(upper[1], upper[0])] = parameters[:21]
    print("Recorded simulator data; assumed zero water current, NED/body frames.")
    print("Target: requested wrench, not measured delivered thrust.")
    print("Constraints: nonnegative linear/quadratic damping; inertia is unconstrained.")
    print(f"Restoring: fully submerged {vehicle} hull, using configured mass/volume/centres.")
    print("Samples (native / training / validation):",
          len(merged["time"]), len(training["time"]), len(validation["time"]))
    print("Regression rank:", rank, "/", len(FOSSEN_PARAMETER_NAMES))
    print(f"Column-scaled condition number: {condition:.3g}")
    with np.printoptions(precision=3, suppress=True):
        print("Estimated total inertia:\n", mass)
        print("Inertia eigenvalues:", np.linalg.eigvalsh(mass))
        print("Linear damping [u,v,w,p,q,r]:", parameters[21:27])
        print("Quadratic damping [u,v,w,p,q,r]:", parameters[27:])
    print("Output units: X/Y/Z in N; K/M/N in N*m. Baseline uses training means.")
    print_evaluation(report)
    if plot or save_plot is not None:
        figure, _ = plot_predictions(
            actual, predicted, output_names, time=validation["time"],
            units=["N"] * 3 + ["N*m"] * 3, save_path=save_plot, show=plot,
        )
        import matplotlib.pyplot as plt
        plt.close(figure)
    if rank < len(FOSSEN_PARAMETER_NAMES):
        print("WARNING: This recording cannot identify all 33 parameters.")
    if np.any(np.linalg.eigvalsh(mass) <= 0):
        print("WARNING: Estimated inertia is not positive definite.")
    return parameters, rmse


def test_sim_full_fossen():
    if any(not (DATA_DIRECTORY / name).is_file()
           for name in ("odom_test.csv", "wrench_input_test.csv")):
        pytest.skip("Requires local simulator odometry and wrench recordings.")
    parameters, rmse = run_sim_fit()
    # Passing checks execution, not model accuracy on these recordings.
    assert parameters.shape == (33,)
    assert rmse.shape == (6,)
    assert np.all(np.isfinite(parameters)) and np.all(np.isfinite(rmse))
    assert np.all(parameters[21:] >= 0)


def test_segment_derivative_and_restoring_convention():
    time = np.array([0.0, 0.2, 0.5, 0.9, 1.2])
    velocity = np.zeros((5, 6))
    velocity[:, 0] = time ** 2
    segment = {
        "time": time,
        "velocity": velocity,
        "orientation": np.tile([0.0, 0.0, 0.0, 1.0], (5, 1)),
        "force": np.zeros((5, 6)),
        "depth": np.ones(5),
    }
    prepared = _prepare_segment(segment)
    np.testing.assert_array_equal(prepared["time"], time[1:-1])
    np.testing.assert_allclose(prepared["acceleration"][:, 0], 2 * time[1:-1])
    expected_g = np.zeros((3, 6))
    expected_g[:, 2] = (WATER_DENSITY * DISPLACED_VOLUME - MASS_KG) * 9.81
    np.testing.assert_allclose(prepared["restoring"], expected_g)
    np.testing.assert_array_equal(segment["velocity"], velocity)

    # +90 degrees roll: world down is sensor +y; CG offset creates roll moment.
    segment["orientation"][:] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    tilted = _prepare_segment(segment)
    expected_g[:] = 0
    expected_g[:, 1] = (WATER_DENSITY * DISPLACED_VOLUME - MASS_KG) * 9.81
    expected_g[:, 3] = CENTER_OF_GRAVITY[2] * MASS_KG * 9.81
    np.testing.assert_allclose(tilted["restoring"], expected_g, atol=1e-10)

    scaled = _prepare_segment(segment, vehicle="sysid")
    expected_g[:, :3] *= 0.6**3
    expected_g[:, 3:] *= 0.6**4
    np.testing.assert_allclose(scaled["restoring"], expected_g, atol=1e-10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--vehicle", choices=VEHICLES, default="sysid")
    parser.add_argument("--plot", action="store_true", help="Show actual/predicted validation outputs.")
    parser.add_argument("--save-plot", type=Path, help="Save the validation plot, e.g. results/fit.png.")
    args = parser.parse_args()
    run_sim_fit(args.data_dir, args.vehicle, plot=args.plot, save_plot=args.save_plot)
