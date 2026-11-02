"""Check metrics, validation and plot contents independently of a fitted model."""
import numpy as np
import pytest

from system_identification.src.evaluate import (
    evaluate, evaluate_parameters, plot_predictions, print_evaluation, rmse,
)


def test_metrics_and_per_output_results(capsys):
    actual = np.array([[0, 1], [1, 2], [2, 3]])
    predicted = actual + [1, 0]
    report = evaluate(actual, predicted, ["u", "v"])
    assert report["u"] == {"r2": -0.5, "rmse": 1, "mae": 1, "bias": 1, "max_error": 1}
    assert report["v"] == {"r2": 1, "rmse": 0, "mae": 0, "bias": 0, "max_error": 0}
    print_evaluation(report)
    assert "max_error" in capsys.readouterr().out


def test_degenerate_r2_and_custom_metrics():
    assert np.isnan(evaluate([2, 2], [2, 2])["output_1"]["r2"])
    assert np.isnan(evaluate([1], [2])["output_1"]["r2"])
    scores = evaluate([0, 1], [1, 2], ["x"], metrics={"my_rmse": rmse})
    assert scores == {"x": {"my_rmse": 1.0}}
    with pytest.raises(ValueError, match="one value"):
        evaluate([0, 1], [1, 2], metrics={"bad": lambda a, b: 0})


@pytest.mark.parametrize("actual,predicted,names", [
    ([], [], None), ([1, 2], [1], None), ([np.nan], [1], None),
    ([1], [np.inf], None), ([1j], [1], None),
    ([[1, 2]], [[1, 2]], ["same", "same"]), ([1, 2], [1, 2], ["a", "b"]),
])
def test_bad_inputs(actual, predicted, names):
    with pytest.raises(ValueError):
        evaluate(actual, predicted, names)


def test_parameter_comparison():
    report = evaluate_parameters([2, 0], [3, -1], ["mass", "coupling"])
    assert report["mass"]["relative_error_percent"] == 50
    assert report["coupling"]["error"] == -1
    assert np.isnan(report["coupling"]["relative_error_percent"])


def test_plot_contents_and_save(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    actual = np.array([[1, 3], [2, 4], [3, 5]])
    predicted = actual + 0.5
    path = tmp_path / "plots" / "fit.png"
    fig, axes = plot_predictions(actual, predicted, ["u", "v"],
                                 time=[10, 11, 13], units=["m/s", "m/s"], save_path=path)
    try:
        assert len(axes) == 2
        np.testing.assert_array_equal(axes[1].lines[0].get_ydata(), actual[:, 1])
        np.testing.assert_array_equal(axes[1].lines[1].get_ydata(), predicted[:, 1])
        np.testing.assert_array_equal(axes[0].lines[0].get_xdata(), [0, 1, 3])
        assert axes[0].get_ylabel() == "u [m/s]"
        assert path.stat().st_size > 0
    finally:
        plt.close(fig)
    fig, axes = plot_predictions([1, 2], [1, 2])
    assert len(axes) == 1
    plt.close(fig)
    with pytest.raises(ValueError, match="Time"):
        plot_predictions([1, 2], [1, 2], time=[0, 0])


def test_trace_dynamics_decay_and_state_selection():
    from system_identification.src.evaluate import trace_dynamics
    time = np.linspace(0, 2, 31)
    decay = trace_dynamics([2], lambda x, theta: -theta * x, time, theta=0.7)
    np.testing.assert_allclose(decay[:, 0], 2 * np.exp(-0.7 * time), rtol=2e-6)
    # Selecting position must still integrate the coupled velocity state.
    position = trace_dynamics([1, 0], lambda x, _: [x[1], -x[0]], time, states=[0])
    assert position.shape == (31, 1)
    np.testing.assert_allclose(position[:, 0], np.cos(time), atol=1e-6)


def test_trace_dynamics_forcing_and_single_sample():
    from system_identification.src.evaluate import trace_dynamics
    time = np.linspace(2, 3, 11)
    f = lambda x, mass, u: np.asarray(u) / mass
    constant = trace_dynamics([0], f, time, theta=2, input_forces=[4])
    np.testing.assert_allclose(constant[:, 0], 2 * (time - 2), atol=1e-10)
    changing = trace_dynamics([0], f, time, theta=2, input_forces=lambda t: [2*t])
    np.testing.assert_allclose(changing[:, 0], (time**2 - 4) / 2, atol=1e-10)
    np.testing.assert_array_equal(trace_dynamics([1, 2], f, [0], states=[1]), [[2]])


@pytest.mark.parametrize("kwargs", [
    {"time": [0, 0]}, {"time": []}, {"start_state": [np.nan]},
    {"states": [2]}, {"states": [0.5]}, {"f": lambda x, _: [1, 2]},
    {"f": lambda x, _: [np.inf]},
])
def test_trace_dynamics_invalid_inputs(kwargs):
    from system_identification.src.evaluate import trace_dynamics
    args = dict(start_state=[1], f=lambda x, _: -x, time=[0, 1])
    args.update(kwargs)
    with pytest.raises(ValueError):
        trace_dynamics(**args)


def test_plot_fossen_resets_and_sampled_commands(monkeypatch):
    import matplotlib.pyplot as plt
    from system_identification.src.evaluate import plot_fossen_model_dynamics
    from system_identification.src.fossen_model import FossenState

    monkeypatch.setattr(plt, "show", lambda: None)
    time = np.arange(5, dtype=float) + 100
    measured = [FossenState([0, 0, 100], [0, 0, 0, 1], [0]*6) for _ in time]
    forces = np.zeros((5, 6))
    forces[:, 0] = [1, 3, 2, 4, 0]

    def f(state, theta, force):
        assert isinstance(state, FossenState)
        dx = np.zeros(13)
        dx[0] = force[0]
        return dx

    predicted, fig, axes = plot_fossen_model_dynamics(
        dict(time=time, state=measured, force=forces), f, reset_time=2,
        states_to_plot=[0],
    )
    try:
        np.testing.assert_allclose(predicted[:, 0], [0, 1, 0, 2, 0], atol=1e-10)
        # Keep the prediction at each reset boundary visible in the plot.
        np.testing.assert_allclose(axes[0].lines[1].get_ydata(), [0, 1, 4], atol=1e-10)
        np.testing.assert_allclose(axes[0].lines[2].get_ydata(), [0, 2, 6], atol=1e-10)
        np.testing.assert_array_equal(measured[0].position, [0, 0, 100])
    finally:
        plt.close(fig)


def test_plot_fossen_no_reset_and_force_override(monkeypatch):
    import matplotlib.pyplot as plt
    from system_identification.src.evaluate import plot_fossen_model_dynamics
    from system_identification.src.fossen_model import FossenState

    monkeypatch.setattr(plt, "show", lambda: None)
    time = np.array([2., 2.5, 3.])
    states = [FossenState([0, 0, 100], [0, 0, 0, 1], [0]*6) for _ in time]

    def f(state, theta, force):
        return np.r_[force[0], np.zeros(12)]

    data = dict(time=time, state=states)
    predicted, fig, axes = plot_fossen_model_dynamics(
        data, f, reset_time=None, input_forces=lambda t: [t, 0, 0, 0, 0, 0],
    )
    try:
        np.testing.assert_allclose(predicted[:, 0], (time**2 - 4)/2, atol=1e-10)
        assert len(axes) == 13
    finally:
        plt.close(fig)
    with pytest.raises(ValueError, match="reset_time"):
        plot_fossen_model_dynamics(data, f, reset_time=0)
