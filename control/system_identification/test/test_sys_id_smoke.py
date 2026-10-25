"""Check the three-parameter model with small, exact data and no files."""

import numpy as np

from system_identification.src.system_identification import SystemIdentification


def test_fit_predict_and_score():
    # F = 2*v + 3*v*abs(v) + 60*acceleration.
    # Effective mass could represent 50 kg physical mass + 10 kg added mass.
    data = {
        "velocity": np.array([-2., -1., 0., 1., 2.]),
        "acceleration": np.array([0., 1., 2., -1., 0.]),
        "force": np.array([-16., 55., 120., -55., 16.]),
    }
    model = SystemIdentification()

    parameters = model.fit_parameters(data)
    predictions = model.predict(data)
    error = model.score(data)

    np.testing.assert_allclose(parameters, [2., 3., 60.], atol=1e-10)
    np.testing.assert_allclose(predictions, data["force"], atol=1e-10)
    assert error < 1e-10

