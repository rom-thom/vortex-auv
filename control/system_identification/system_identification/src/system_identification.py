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

    def build_regression_1D(self, data):
        """solves  d₁*v + d₂*v|v| = F - m·dv/dt  for the d values, asumes known mass and force
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
        if not all(np.all(np.isfinite(values)) for values in (velocity, acceleration, force)) or not np.isfinite(mass):
            raise ValueError("All data values must be finite.")

        # d1*v + d2*v*|v| + mass * acceleration = force
        A = np.column_stack((velocity, velocity * np.abs(velocity), acceleration))
        b = force

        self.regression_matrix = A
        self.data_vector = b
        return A, b


        

    # Generelt for alle system:


    # The big the bad the ugly (build regresion), dependent on model type (gray box)
    def build_regression(self, data):
        return self.build_regression_1D(data)

    # A * x = b <- solves for x
    def fit_parameters(self, data):
        A, b = self.build_regression(data)
        parameters, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.parameters = parameters
        return parameters


    def predict(self, data):
        if self.parameters is None:
            raise ValueError("The model needs fitting before prediction.")

        A, _ = self.build_regression(data)
        return A @ self.parameters

    # Calculates the two norm of each error
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

        arrays = {name: np.asarray(value) for name, value in data.items()} # To make shure they have the correct structure
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
        "force": np.array([0.0, 12.0, 18.0, 10.0, 3.0])
    }
