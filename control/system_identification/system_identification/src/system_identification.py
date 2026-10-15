import numpy as np




class SystemIdentification:
    def __init__(self):
        self.parameters = None
        self.regression_matrix = None
        self.data_vector = None

    def least_squares(self, A, b):
        """Solve A @ x = b by least squares and return the estimated parameters."""
        parameters, *_ = np.linalg.lstsq(A, b, rcond=None)
        return parameters

    def build_regression_simple_1D(self, data):
        """solves  d₁*v + d₂*v|v| = F - m·dv/dt  for the d values, asumes known mass and force
        data = {
            "velocity":     np.array([0.1, 0.2]),     # m/s
            "acceleration": np.array([0.5, 0.4]),     # m/s²
            "force":        np.array([28.0, 35.0]),   # N
            "mass": 53.7,                             # kg
        }"""

        velocity = np.asarray(data["velocity"], dtype=float).reshape(-1)
        acceleration = np.asarray(data["acceleration"], dtype=float).reshape(-1)
        force = np.asarray(data["force"], dtype=float).reshape(-1)
        mass = float(data["mass"])

        if velocity.size == 0:
            raise ValueError("At least one sample is required.")
        if not (velocity.shape == acceleration.shape == force.shape):
            raise ValueError(
                "velocity, acceleration, and force must have matching lengths."
            )
        if not all(np.all(np.isfinite(values)) for values in (velocity, acceleration, force)) or not np.isfinite(mass):
            raise ValueError("All data values must be finite.")

        # d1*v + d2*v*|v| = force - mass*acceleration
        A = np.column_stack((velocity, velocity * np.abs(velocity)))
        b = force - mass * acceleration

        self.regression_matrix = A
        self.data_vector = b
        return A, b


        

    # Generelt for alle system:


    # The big the bad the ugly (build regresion), dependent on model type (gray box)
    def build_regression(self, data):
        return self.build_regression_simple_1D(data)

    # A * x = b <- solves for x
    def fit_parameters(self, data):
        A, b = self.build_regression(data)
        parameters, *_ = np.linalg.lstsq(A, b, rcond=None)
        return parameters


    def predict(self, data):
        if self.parameters is None:
            raise ValueError("The model needs fitting before prediction.")

        A, _ = self.build_regression(data)
        return A @ self.parameters

    def score(self, data):
        if self.parameters is None:
            raise ValueError("The model needs to be fitted before scoring.")
        A, b = self.build_regression(data)
        predictions = A @ self.parameters
        return np.sqrt(np.mean((b - predictions) ** 2))


    def split_data(self, data: dict, training_fraction: float) -> tuple[dict, dict]:
        """
        Return training and validation data without shuffling sample order.
        The first floor(n * training_fraction) samples become training
        data; the remainder become validation data.
        """
        if not 0 < training_fraction < 1:
            raise ValueError("training_fraction must be strictly between 0 and 1.")

        arrays = {name: np.asarray(value) for name, value in data.items()} # Make shure thay are dictionaries
        lengths = {len(values) for values in arrays.values() if values.ndim > 0}
        if not lengths:
            raise ValueError("Data must contain at least one sample array.")
        if len(lengths) != 1:
            raise ValueError("All sample arrays must have matching lengths.")

        sample_count = lengths.pop()
        split_index = int(sample_count * training_fraction)
        if not 0 < split_index < sample_count:
            raise ValueError("The split must leave at least one sample in each dataset.")

        training, validation = {}, {}
        for name, values in arrays.items():
            if values.ndim == 0:
                training[name] = validation[name] = values.item()
            else:
                training[name] = values[:split_index].copy()
                validation[name] = values[split_index:].copy()
        return training, validation













if __name__ == "__main__":
    model = SystemIdentification()
    data = {
        "time": np.arange(5, dtype=float),
        "velocity": [0.0, 0.2, 0.4, 0.3, 0.1],
        "force": np.array([0.0, 12.0, 18.0, 10.0, 3.0]),
        "mass": 53.7,
    }

    # Explicit sizes also check rounding down for an odd number of samples.
    for fraction, expected_count in ((0.2, 1), (0.5, 2), (0.8, 4)):
        training, validation = model.split_data(data, fraction)
        np.testing.assert_equal(set(training), set(data))
        np.testing.assert_equal(set(validation), set(data))
        np.testing.assert_equal(training["mass"], data["mass"])
        np.testing.assert_equal(validation["mass"], data["mass"])

        for name in ("time", "velocity", "force"):
            original = np.array(data[name], copy=True)
            np.testing.assert_array_equal(
                training[name], original[:expected_count]
            )
            np.testing.assert_array_equal(
                validation[name], original[expected_count:]
            )

            # Editing either split must leave the input and the other split intact.
            training[name][0] = -100.0
            np.testing.assert_array_equal(
                validation[name], original[expected_count:]
            )
            validation[name][0] = -200.0
            np.testing.assert_equal(training[name][0], -100.0)
            np.testing.assert_array_equal(data[name], original)

        print(f"PASS: fraction={fraction}: {expected_count} training, "
              f"{len(data['time']) - expected_count} validation samples")

    for fraction in (0.0, 1.0, -0.1, 1.1, np.nan, np.inf):
        with np.testing.assert_raises_regex(ValueError, "training_fraction"):
            model.split_data(data, fraction)

    invalid_cases = (
        ({}, 0.5, "sample array"),
        ({"mass": 53.7}, 0.5, "sample array"),
        ({"velocity": [0.1, 0.2], "force": [1.0]}, 0.5, "matching lengths"),
        ({"velocity": []}, 0.5, "at least one sample"),
        ({"velocity": [0.1]}, 0.5, "at least one sample"),
        ({"velocity": [0.1, 0.2]}, 0.1, "at least one sample"),
    )
    for invalid_data, fraction, message in invalid_cases:
        with np.testing.assert_raises_regex(ValueError, message):
            model.split_data(invalid_data, fraction)

    print("PASS: invalid fractions and unusable datasets raise ValueError")
    print("All split_data checks passed.")
