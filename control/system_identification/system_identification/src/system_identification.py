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
    

