"""Model-independent prediction metrics and plots, with one result per output.

Inputs are aligned arrays shaped (samples, outputs), or (samples,) for one
output. R² describes output predictions, not individual model parameters.
"""

from pathlib import Path

import numpy as np

if __package__:
    from .the_legend_fossen.data_classes import FossenState
else:
    from the_legend_fossen.data_classes import FossenState

    

def _array(values, name):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real.")
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError(f"{name} must have nonempty shape (samples, outputs) or (samples,).")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _paired_data(actual, predicted):
    actual, predicted = _array(actual, "actual"), _array(predicted, "predicted")
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted data must have matching shapes.")
    return actual, predicted


def _names(output_names, count):
    names = [f"output_{i + 1}" for i in range(count)] if output_names is None else list(output_names)
    if len(names) != count or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("Supply one nonempty output name per column.")
    if len(set(names)) != len(names):
        raise ValueError("Output names must be unique.")
    return names


def residuals(actual, predicted):
    """Prediction minus observation, shaped (samples, outputs)."""
    actual, predicted = _paired_data(actual, predicted)
    return predicted - actual


def rmse(actual, predicted):
    """Root mean squared error per output, in that output's units."""
    return np.sqrt(np.mean(residuals(actual, predicted) ** 2, axis=0))


def mae(actual, predicted):
    """Mean absolute error per output, in that output's units."""
    return np.mean(np.abs(residuals(actual, predicted)), axis=0)


def bias(actual, predicted):
    """Mean signed error per output; positive means overprediction."""
    return np.mean(residuals(actual, predicted), axis=0)


def max_error(actual, predicted):
    """Maximum absolute error per output."""
    return np.max(np.abs(residuals(actual, predicted)), axis=0)


def r2(actual, predicted):
    """1 - SSE/SST per output. Constant targets or fewer than 2 samples give NaN.

    One is perfect, zero matches the evaluation-set mean, negative is worse.
    """
    actual, predicted = _paired_data(actual, predicted)
    result = np.full(actual.shape[1], np.nan)
    if len(actual) < 2:
        return result
    sst = np.sum((actual - actual.mean(axis=0)) ** 2, axis=0)
    valid = np.any(actual != actual[0], axis=0) & (sst > 0)
    sse = np.sum((predicted - actual) ** 2, axis=0)
    np.divide(sse, sst, out=result, where=valid)
    return 1 - result


DEFAULT_METRICS = {"r2": r2, "rmse": rmse, "mae": mae, "bias": bias, "max_error": max_error}


def evaluate(actual_data, predicted_data, output_names=None, *, metrics=None):
    """Return {output_name: {metric_name: float}} for any prediction model.

    Pass a metric-name -> function mapping to select/add metrics. Each function
    takes actual/predicted (N, D) arrays and returns D scores. Use aligned,
    held-out data in the same units/frames. No fitting or resampling is done.
    """
    actual, predicted = _paired_data(actual_data, predicted_data)
    names = _names(output_names, actual.shape[1])
    report = {name: {} for name in names}
    for metric_name, function in (DEFAULT_METRICS if metrics is None else metrics).items():
        scores = np.asarray(function(actual, predicted), dtype=float)
        if scores.shape != (len(names),):
            raise ValueError(f"Metric {metric_name!r} must return one value per output.")
        for name, score in zip(names, scores):
            report[name][metric_name] = float(score)
    return report


def print_evaluation(report):
    """Print the dictionary returned by evaluate as a compact table."""
    if not report:
        return
    metrics = list(next(iter(report.values())))
    width = max(12, *(len(name) for name in report))
    print(f"{'Output':<{width}} " + " ".join(f"{m:>12}" for m in metrics))
    for name, values in report.items():
        print(f"{name:<{width}} " + " ".join(f"{values[m]:12.4g}" for m in metrics))


def evaluate_parameters(actual, estimated, parameter_names=None):
    """Compare parameter vectors when ground truth is known; not an R² score.

    Returns signed/absolute errors and absolute relative error in percent.
    Relative error is NaN for a zero reference parameter.
    """
    actual, estimated = np.asarray(actual), np.asarray(estimated)
    if actual.ndim != 1 or estimated.ndim != 1:
        raise ValueError("Parameters must be one-dimensional vectors.")
    actual, estimated = _paired_data(actual, estimated)
    actual, estimated = actual[:, 0], estimated[:, 0]
    names = _names(parameter_names, len(actual))
    error = estimated - actual
    relative = np.full(len(actual), np.nan)
    np.divide(100 * np.abs(error), np.abs(actual), out=relative, where=actual != 0)
    return {name: {"actual": float(a), "estimated": float(e), "error": float(d),
                   "absolute_error": float(abs(d)), "relative_error_percent": float(r)}
            for name, a, e, d, r in zip(names, actual, estimated, error, relative)}


def plot_predictions(actual_data, predicted_data, output_names=None, *, time=None,
                     units=None, save_path=None, show=False):
    """Plot observed/predicted outputs on separate axes; return (figure, axes).

    time is an optional increasing (N,) array in seconds. Otherwise use sample
    indices. units is an optional label per output. Save with save_path and/or
    display with show=True. Works equally for states, forces and other outputs.
    """
    actual, predicted = _paired_data(actual_data, predicted_data)
    names = _names(output_names, actual.shape[1])
    units = [""] * len(names) if units is None else list(units)
    if len(units) != len(names):
        raise ValueError("Supply one unit label per output.")
    x = np.arange(len(actual))
    xlabel = "Sample"
    if time is not None:
        if np.iscomplexobj(time):
            raise ValueError("Time must be real.")
        x = np.asarray(time, dtype=float)
        if x.shape != (len(actual),) or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
            raise ValueError("Time must be finite, strictly increasing and match the sample count.")
        x = x - x[0]
        xlabel = "Elapsed time [s]"

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(names), 1, sharex=True, squeeze=False,
                             figsize=(10, max(3, 2.4 * len(names))), constrained_layout=True)
    axes = axes[:, 0]
    for i, (ax, name, unit) in enumerate(zip(axes, names, units)):
        ax.plot(x, actual[:, i], label="Actual", linewidth=1.3)
        ax.plot(x, predicted[:, i], label="Predicted", linestyle="--", linewidth=1.1)
        ax.set_ylabel(f"{name} [{unit}]" if unit else name)
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel(xlabel)
    if save_path is not None:
        path = Path(save_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig, axes








def trace_dynamics(start_state: FossenState, f, time, theta=None, input_forces=None,
                   states=None, method="RK45", rtol=1e-6, atol=1e-9,
                   max_step=np.inf):
        
    """integrates the state with our estimated model

    x_dot = f(x, theta, input_forces)
    state is what states we are looking at as there are quite a lot of them (None means we look at all)
    A FossenState start passes FossenState objects to f; array starts pass arrays.
    Returns a numeric array (samples, selected states) for evaluation/plotting.
    """
    from scipy.integrate import solve_ivp

    if np.iscomplexobj(time):
        raise ValueError("time must be real.")
    use_state = isinstance(start_state, FossenState)
    initial_values = start_state.as_array() if use_state else start_state
    if np.iscomplexobj(initial_values):
        raise ValueError("start_state must be real.")
    initial = np.asarray(initial_values, dtype=float)
    if initial.ndim != 1 or initial.size == 0 or not np.all(np.isfinite(initial)):
        raise ValueError("start_state must be a nonempty finite 1-D array or FossenState.")
    times = np.asarray(time, dtype=float)
    if (times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("time must be a nonempty finite, strictly increasing 1-D array.")
    if not callable(f):
        raise TypeError("f must be callable.")

    
    indices = np.arange(initial.size)
    if states is not None:
        indices = np.asarray(states)
        if (indices.ndim != 1 or indices.size == 0
                or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= initial.size)):
            raise ValueError("states must be a nonempty list of valid state indices.")

    def derivative(t, x):
        """This is only for finding dx = f(x, theta), however chat chat was like:⠀
           "EEhhrm, we should make it complicated you know"
           """
        
        if use_state:
            x = FossenState.from_array(x)
        if input_forces is None:
            dx = f(x, theta)
        else:
            forces = input_forces(t) if callable(input_forces) else input_forces
            dx = f(x, theta, forces)
        if np.iscomplexobj(dx):
            raise ValueError("f must return real state derivatives.")
        dx = np.asarray(dx, dtype=float)
        if dx.shape != initial.shape or not np.all(np.isfinite(dx)):
            raise ValueError("f must return finite derivatives matching start_state's shape.")
        return dx

    if len(times) == 1:
        return initial[None, indices].copy()
    solution = solve_ivp(
        derivative, (times[0], times[-1]), initial, t_eval=times,
        method=method, rtol=rtol, atol=atol, max_step=max_step,
    )
    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")
    return solution.y.T[:, indices]


import matplotlib.pyplot as plt

def plot_fossen_model_dynamics(data, f, reset_time, theta=None, input_forces=None,
                   states_to_plot=None, method="RK45", rtol=1e-6, atol=1e-9, max_step=np.inf):
    """Plot measured and simulated states, resetting from measurements periodically.

    data: time (N,), state (N FossenState objects), force (N, 6).
    reset_time is seconds between resets (None disables resets).
    Recorded forces are held constant until the next sample. input_forces: can
    override them with a constant six-vector or a function of absolute time.
    states_to_plot: selects indices 0..12; None plots all states.
    Returns (predicted, figure, axes), with predicted shaped (N, 13). Reset
    rows contain the measured state; lines also show the pre-reset prediction.
    """
    if np.iscomplexobj(data["time"]):
        raise ValueError("time must be real.")
    time = np.asarray(data["time"], dtype=float)
    if (time.ndim != 1 or time.size == 0 or not np.all(np.isfinite(time))
            or np.any(np.diff(time) <= 0)):
        raise ValueError("time must be finite, nonempty and strictly increasing.")
    if reset_time is not None and (not np.isfinite(reset_time) or reset_time <= 0):
        raise ValueError("reset_time must be positive or None.")
    measured = list(data["state"])
    if len(measured) != len(time) or not all(isinstance(x, FossenState) for x in measured):
        raise ValueError("state must contain one FossenState per timestamp.")
    actual = _array([x.as_array() for x in measured], "state")
    if actual.shape != (len(time), 13):
        raise ValueError("Each state must have 13 values.")
    indices = np.arange(13) if states_to_plot is None else np.asarray(states_to_plot)
    if (indices.ndim != 1 or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= 13)):
        raise ValueError("states_to_plot must contain state indices from 0 to 12.")
    forces = None
    if input_forces is None:
        forces = _array(data["force"], "force")
        if forces.shape != (len(time), 6):
            raise ValueError("force must have shape (N, 6).")

    starts = [0]
    for i in range(1, len(time)):
        if reset_time is not None and time[i] - time[starts[-1]] >= reset_time:
            starts.append(i)
    segments = []
    predicted = np.empty_like(actual)
    options = dict(method=method, rtol=rtol, atol=atol, max_step=max_step)
    for start, end in zip(starts, starts[1:] + [len(time) - 1]):
        segment_time = time[start:end + 1]
        if forces is None:
            segment = trace_dynamics(measured[start], f, segment_time, theta,
                                     input_forces=input_forces, **options)
        else:
            # Integrate separately at each command change so steps cannot be skipped.
            segment = np.empty((len(segment_time), 13))
            segment[0] = actual[start]
            for j in range(start, end):
                step = trace_dynamics(
                    FossenState.from_array(segment[j - start]), f, time[j:j + 2],
                    theta, input_forces=forces[j], **options,
                )
                segment[j - start + 1] = step[-1]
        predicted[start:end + 1] = segment
        segments.append((segment_time, segment))

    names = ["north", "east", "down", "qx", "qy", "qz", "qw",
             "u", "v", "w", "p", "q", "r"]
    units = ["m"] * 3 + [""] * 4 + ["m/s"] * 3 + ["rad/s"] * 3
    fig, axes = plt.subplots(len(indices), 1, sharex=True, squeeze=False,
                             figsize=(10, max(3, 2.4 * len(indices))), constrained_layout=True)
    axes = axes[:, 0]
    for ax, index in zip(axes, indices):
        ax.plot(time - time[0], actual[:, index], label="Actual")
        for k, (segment_time, segment) in enumerate(segments):
            ax.plot(segment_time - time[0], segment[:, index], "--", color="tab:orange",
                    marker="." if len(segment_time) == 1 else None,
                    label="Predicted" if k == 0 else None)
        ax.set_ylabel(f"{names[index]} [{units[index]}]" if units[index] else names[index])
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Elapsed time [s]")
    plt.show()
    return predicted, fig, axes


if __name__ == "__main__":
    from functools import partial
    if __package__:
        from .data_handeling import load_data
        from .the_legend_fossen.fossen_model import fossen_dynamics
    else:
        from data_handeling import load_data
        from the_legend_fossen.fossen_model import fossen_dynamics

    directory = Path(__file__).resolve().parents[1] / "data" / "sysid"
    odom = load_data(directory / "odom_test.csv")
    wrench = load_data(directory / "wrench_input_test.csv")

    # Use 30 seconds with both odometry and force commands available.
    start = max(odom["time"][0], wrench["time"][0])
    end = min(start + 30, odom["time"][-1], wrench["time"][-1])
    keep = (odom["time"] >= start) & (odom["time"] <= end)
    time = odom["time"][keep]
    position = np.column_stack([odom[f"pose.pose.position.{a}"][keep] for a in "xyz"])
    orientation = np.column_stack([odom[f"pose.pose.orientation.{a}"][keep] for a in "xyzw"])
    velocity = np.column_stack([
        odom[f"twist.twist.{kind}.{a}"][keep]
        for kind in ("linear", "angular") for a in "xyz"
    ])
    states = [FossenState(p, q, v) for p, q, v in zip(position, orientation, velocity)]
    forces = np.column_stack([
        wrench[f"wrench.{kind}.{a}"] for kind in ("force", "torque") for a in "xyz"
    ])

    def recorded_forces(t):
        index = np.searchsorted(wrench["time"], t, side="right") - 1
        return forces[np.clip(index, 0, len(forces) - 1)]

    # Rough example parameters, not fitted: inertia, linear and quadratic damping.
    M = np.diag([190., 190., 200., 1.2, 10., 5.])
    theta = np.r_[M[np.triu_indices(6)], [250, 100, 300, 5, 8, 10], [0, 200, 500, 0, 0, 0]]
    dynamics = partial(fossen_dynamics, mass_kg=11.5992, volume=0.0108320817855168,
                       center_of_gravity=[0, 0, 0.006])
    data = {"time": time, "state": states}
    predicted, fig, axes = plot_fossen_model_dynamics(
        data, dynamics, reset_time=5, theta=theta, input_forces=recorded_forces,
        states_to_plot=[7, 8, 9], max_step=0.05,  # u, v, w
    )
