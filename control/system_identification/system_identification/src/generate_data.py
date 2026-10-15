"""Generate a reproducible synthetic experiment for the 1-D damping model."""

import argparse
import json
from pathlib import Path

import numpy as np


MASS = 53.7  # kg; matches config/config.yaml
D1 = 12.0  # N s/m; illustrative, not measured Nautilus damping
D2 = 28.0  # N s^2/m^2; illustrative, not measured Nautilus damping
DT = 0.05  # 20 Hz, synchronized samples
HOLD_SECONDS = 20.0
THRUST_LEVELS = [0, 8, 25, 60, 15, 0, -10, -35, -65, -15, 0, 40, -40, 10, 0]
THRUST_TIME_CONSTANT = 0.4  # s
THRUST_SLEW_LIMIT = 2.0  # N/s for the total 1-D force in this synthetic model
DISTURBANCE_STD = 0.6  # N
DISTURBANCE_TIME_CONSTANT = 2.0  # s
NOISE_STD = {"velocity": 0.008, "acceleration": 0.012, "force": 0.4}
BIAS = {"velocity": 0.002, "acceleration": 0.0015, "force": 0.15}


def generate_experiment(seed=42):
    """Return measured columns, ground truth, and generation settings.

    Integrate m*dv/dt = thrust - d1*v - d2*v*abs(v) + disturbance.
    Sensor noise is added after integration; force is an independent input.
    """
    rng = np.random.default_rng(seed)
    duration = len(THRUST_LEVELS) * HOLD_SECONDS
    time = np.arange(round(duration / DT) + 1) * DT
    stage = np.minimum((time / HOLD_SECONDS).astype(int), len(THRUST_LEVELS) - 1)
    command = np.asarray(THRUST_LEVELS, dtype=float)[stage]

    # Stationary, correlated external force, distinct from applied thrust.
    rho = np.exp(-DT / DISTURBANCE_TIME_CONSTANT)
    disturbance = np.empty(time.size)
    disturbance[0] = rng.normal(0.0, DISTURBANCE_STD)
    for i in range(1, time.size):
        disturbance[i] = (
            rho * disturbance[i - 1]
            + DISTURBANCE_STD * np.sqrt(1.0 - rho**2) * rng.normal()
        )

    # State is [velocity, applied thrust]. RK4 integrates actuator and vehicle.
    state = np.zeros((time.size, 2))

    def derivative(current, target, external):
        velocity, thrust = current
        acceleration = (
            thrust - D1 * velocity - D2 * velocity * abs(velocity) + external
        ) / MASS
        thrust_rate = np.clip(
            (target - thrust) / THRUST_TIME_CONSTANT,
            -THRUST_SLEW_LIMIT,
            THRUST_SLEW_LIMIT,
        )
        return np.array([acceleration, thrust_rate])

    for i in range(time.size - 1):
        current = state[i]
        target = command[i]
        middle_disturbance = (disturbance[i] + disturbance[i + 1]) / 2.0
        k1 = derivative(current, target, disturbance[i])
        k2 = derivative(current + DT * k1 / 2, target, middle_disturbance)
        k3 = derivative(current + DT * k2 / 2, target, middle_disturbance)
        k4 = derivative(current + DT * k3, target, disturbance[i + 1])
        state[i + 1] = current + DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    velocity, force = state.T
    acceleration = (
        force - D1 * velocity - D2 * velocity * np.abs(velocity) + disturbance
    ) / MASS
    truth = {
        "time": time,
        "velocity": velocity,
        "acceleration": acceleration,
        "force": force,
        "force_command": command,
        "disturbance_force": disturbance,
    }
    measured = {"time": time.copy()}
    for name in NOISE_STD:
        measured[name] = (
            truth[name] + BIAS[name] + rng.normal(0, NOISE_STD[name], time.size)
        )

    metadata = {
        "synthetic": True,
        "seed": seed,
        "numpy_version": np.__version__,
        "equation": (
            "mass * acceleration + d1 * velocity + d2 * velocity * abs(velocity)"
            " = force + disturbance_force"
        ),
        "mass": MASS,
        "d1": D1,
        "d2": D2,
        "units": {
            "time": "s", "velocity": "m/s", "acceleration": "m/s^2",
            "force": "N", "force_command": "N", "disturbance_force": "N",
            "mass": "kg", "d1": "N s/m", "d2": "N s^2/m^2",
        },
        "sample_rate_hz": 1 / DT,
        "duration_seconds": duration,
        "sample_count": time.size,
        "thrust_levels_N": THRUST_LEVELS,
        "hold_seconds": HOLD_SECONDS,
        "thrust_time_constant_seconds": THRUST_TIME_CONSTANT,
        "thrust_slew_limit_N_per_s": THRUST_SLEW_LIMIT,
        "measurement_noise_std": NOISE_STD,
        "measurement_bias": BIAS,
        "disturbance_std_N": DISTURBANCE_STD,
        "disturbance_time_constant_seconds": DISTURBANCE_TIME_CONSTANT,
        "assumptions": [
            "Illustrative damping and sensor errors, not calibrated Nautilus values.",
            "53.7 kg is the entire inertia of this simplified model; no added mass.",
            "One translation axis; no rotation, cross-axis coupling, or water current.",
            "Acceleration is kinematic acceleration, with no gravity component.",
            "All measurements are synchronized; no dropouts or outliers.",
            "Force is noisy applied thrust, not the actuator command or mass * acceleration.",
        ],
    }
    return measured, truth, metadata


def write_csv(path, columns):
    """Save named numeric columns in the format accepted by load_data."""
    np.savetxt(
        path, np.column_stack(list(columns.values())), delimiter=",",
        header=",".join(columns), comments="", fmt="%.10f",
    )


def save_plot(path, measured, truth):
    """Save a static comparison of measurements and ground truth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for axis, name, unit in zip(
        axes, ("force", "velocity", "acceleration"), ("N", "m/s", "m/s²")
    ):
        axis.plot(measured["time"], measured[name], color="tab:blue",
                  linewidth=0.5, alpha=0.6, label="Measured (synthetic)")
        axis.plot(truth["time"], truth[name], color="black",
                  linewidth=1.2, label="Noise-free signal")
        if name == "force":
            axis.plot(truth["time"], truth["force_command"], "--",
                      color="tab:orange", linewidth=1, label="Thrust command")
        axis.set_ylabel(f"{name.capitalize()} [{unit}]")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Synthetic 1-D experiment: m = 53.7 kg, d₁ = 12, d₂ = 28")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    """Write measurements, ground truth, settings, and an optional plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path,
        default=Path("system_identification/data/experiment.csv"),
        help="Measurement CSV destination (overwrites existing generated files).",
    )
    parser.add_argument("--plot", action="store_true", help="Also save a PNG plot.")
    args = parser.parse_args()
    measured, truth, metadata = generate_experiment(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, measured)
    write_csv(args.output.with_name(args.output.stem + "_truth.csv"), truth)
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if args.plot:
        save_plot(args.output.with_suffix(".png"), measured, truth)
    print(f"Saved {metadata['sample_count']} samples to {args.output}")


if __name__ == "__main__":
    main()
