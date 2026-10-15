# Synthetic 1-D identification experiment

`system_identification/data/experiment.csv` contains 6,001 synchronized samples
covering 300 seconds at 20 Hz. Its columns are `time`, `velocity`, `acceleration`,
and `force`, in seconds, m/s, m/s², and N respectively. This replaces the small
`time,values` example with data accepted by the current estimator.

The data is simulated, not measured on Nautilus. Mass is taken from the package's
configuration; damping, actuator dynamics, and noise are illustrative assumptions.
They are not experimentally identified vehicle or sensor specifications.

## Physical model

```text
53.7 * acceleration + 12 * velocity + 28 * velocity * abs(velocity)
    = applied_thrust + external_disturbance
```

The known parameters are mass = 53.7 kg, d₁ = 12 N s/m, and d₂ = 28 N s²/m².
Both damping terms oppose motion. At 1 m/s their combined resistance is 40 N;
linear and quadratic resistance are equal at about 0.43 m/s. The experiment
visits speeds on both sides of that value to help distinguish the coefficients.

Velocity is obtained by integrating the dynamics with RK4, rather than drawing
unrelated random velocity and acceleration samples. Thrust commands change every
20 seconds through positive, negative, and zero levels. Applied thrust has a
0.4-second time constant and a 2 N/s slew limit. The resulting noise-free speed
range is approximately -1.32 to +1.25 m/s, with thrust between -65 and +60 N.

This is a simplified single-axis model. It excludes added mass, rotations,
cross-axis coupling, and water current. Acceleration is kinematic acceleration,
with gravity already excluded. A full marine-craft model includes additional
terms; see [Fossen's model](https://www.fossen.biz/html/marineCraftModel.html).

## Measurement imperfections

Independent Gaussian measurement noise and fixed biases are added after simulation:

| Signal | Noise standard deviation | Fixed bias |
| --- | --- | --- |
| Velocity | 0.008 m/s | +0.002 m/s |
| Acceleration | 0.012 m/s² | +0.0015 m/s² |
| Applied thrust | 0.4 N | +0.15 N |

An additional correlated external force has a standard deviation of 0.6 N and
a correlation time of 2 seconds. It affects the simulated motion but is not
included in the measured thrust. There are no timestamp offsets, dropouts, or
outliers in this baseline dataset. Noisy regressors and sensor biases can bias
ordinary least squares; exact recovery is not expected.

## Load and fit

Run from the package directory using a Python environment with NumPy installed:

```python
import json
from pathlib import Path

from system_identification.src.data_io import load_data
from system_identification.src.system_identification import SystemIdentification

path = Path("system_identification/data/experiment.csv")
data = load_data(path)
settings = json.loads(path.with_suffix(".json").read_text())
data["mass"] = settings["mass"]  # The estimator expects one scalar mass.

parameters = SystemIdentification().fit_parameters(data)
print("Estimated [d1, d2]:", parameters)
```

The default seed gives approximately `[12.19465, 28.04560]`, errors of 1.62%
and 0.16%. The force-balance residual RMS is approximately 1.055 N.
Fitting the first 200 seconds gives `[12.57215, 27.69897]`; evaluation on the
remaining 100 seconds gives a force-balance RMS error of 1.029 N, equivalent
to a one-step acceleration error of 0.0192 m/s². This is not a free-running
trajectory validation or a validation against real vehicle measurements.

## Reproduce and inspect

```bash
python -m system_identification.src.generate_data --plot
```

This recreates the following ignored local artifacts (overwriting existing files):

- `experiment.csv`: noisy measurements for fitting.
- `experiment_truth.csv`: noise-free signals, thrust commands, and disturbances
  for diagnosis. Do not substitute these for measurements in a noisy-data test.
- `experiment.json`: parameters, units, random seed, and assumptions.
- `experiment.png`: plots comparing measurements and truth (`--plot` needs Matplotlib).

Use `--seed 7 --output system_identification/data/experiment_seed7.csv` for another
noise realization with the same thrust program. The generator and this guide can
be committed; the generated data stays ignored by Git.
