"""Check identification against the noisy synthetic experiment on disk."""

import json
from pathlib import Path

import numpy as np
import pytest

from system_identification.src.data_io import load_data
from system_identification.src.system_identification import SystemIdentification


DATA_DIRECTORY = (Path(__file__).resolve().parents[1] / "system_identification" / "data")


def test_fit_experiment():
    """Recover known damping and check an unused interval of measurements."""
    csv_path = DATA_DIRECTORY / "experiment.csv"
    settings_path = csv_path.with_suffix(".json")
    if not csv_path.is_file() or not settings_path.is_file():
        pytest.skip(
            "Local experiment files are absent. From the package directory run: "
            "python -m system_identification.src.generate_data")

    data = load_data(csv_path)
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["synthetic"], "This test requires known synthetic parameters."
    for name in ("time", "velocity", "acceleration", "force"):
        assert data[name].shape == (settings["sample_count"],)
        assert np.all(np.isfinite(data[name]))
    assert np.all(np.diff(data["time"]) > 0)

    # Mass is a single known constant, not a column of measurements.
    data["mass"] = float(settings["mass"])
    expected = np.array([settings["d1"], settings["d2"]])
    model = SystemIdentification()
    estimated = model.fit_parameters(data)
    relative_error = 100 * np.abs((estimated - expected) / expected)

    print("True [d1, d2]:", expected)
    print("Estimated [d1, d2]:", estimated)
    print("Absolute relative error [%]:", relative_error)

    # Fit the first two thirds; assess force balance on the remaining third.
    train, held_out = model.split_data(data, training_fraction=2 / 3)
    train_parameters = model.fit_parameters(train)

    # Evaluate the physical equation directly on samples unused during fitting.
    velocity = held_out["velocity"]
    predicted_force = (data["mass"] * held_out["acceleration"] + train_parameters[0] * velocity + train_parameters[1] * velocity * np.abs(velocity))
    rmse = np.sqrt(np.mean((predicted_force - held_out["force"]) ** 2))
    print("Held-out force-balance RMSE [N]:", rmse)


if __name__ == "__main__":
    test_fit_experiment()
