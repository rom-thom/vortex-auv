import numpy as np
from system_identification.src.system_identification import SystemIdentification

rng = np.random.default_rng(42)
n = 10

# Kjende "sanne" verdiar
mass = 53.7
d1_true = 4.2
d2_true = 0.8

# Simulerte målingar: varier både retning og storleik på farten
velocity = rng.uniform(-2.0, 2.0, n)
acceleration = rng.uniform(-0.5, 0.5, n)

# F = m*v_dot + d1*v + d2*v*|v|
force = (
    mass * acceleration
    + d1_true * velocity
    + d2_true * velocity * np.abs(velocity)
)

data = {
    "velocity": velocity,
    "acceleration": acceleration,
    "force": force,
    "mass": mass,
}

model = SystemIdentification()
A, b = model.build_regression(data)
# estimated = model.fit_parameters(data)
estimated, *_ = np.linalg.lstsq(A, b, rcond=None)

print("Sanne [d1, d2]:", [d1_true, d2_true])
print("Estimerte [d1, d2]:", estimated)
