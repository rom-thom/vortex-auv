# Evaluate predictions

`system_identification.src.evaluate` works with any model that predicts numeric
outputs. Supply aligned arrays `(samples, outputs)`, or `(samples,)` for one
output, in matching units and coordinate frames. A flattened multi-output
prediction must be reshaped explicitly. Use held-out data.

```python
from system_identification.src.evaluate import evaluate, print_evaluation, plot_predictions

# Example: measured and predicted velocities, each shaped (N, 3).
names = ["u", "v", "w"]
report = evaluate(actual_velocity, predicted_velocity, names)
print_evaluation(report)
print(report["u"]["r2"])

figure, axes = plot_predictions(
    actual_velocity, predicted_velocity, names,
    time=time, units=["m/s"] * 3,
    save_path="results/velocity.png", show=True,
)
```

The default metrics are R² (larger is better), RMSE, MAE, signed bias
(prediction minus actual), and maximum absolute error. Error metrics retain
the units of each output. R² is NaN for constant observations or fewer than two
samples, and can be negative. Scores are not averaged across different units.

To add a metric, write a function accepting two `(N, D)` arrays and returning
one value per output `(D,)`, then pass a mapping:

```python
import numpy as np
from system_identification.src.evaluate import DEFAULT_METRICS

def median_absolute_error(actual, predicted):
    return np.median(np.abs(predicted - actual), axis=0)

report = evaluate(actual_velocity, predicted_velocity, names,
                  metrics={**DEFAULT_METRICS, "median_ae": median_absolute_error})
```

R² evaluates predictions, not the reliability of fitted parameters. If true
parameter values are known, use `evaluate_parameters(true_values, estimates,
parameter_names)` for signed, absolute and relative errors. Relative error is
NaN when a true parameter is zero. This does not estimate parameter uncertainty
or identify which terms are necessary.

# Existing simulator diagnostic

```bash
python test/test_sim_full_fossen.py --vehicle sysid --plot
python test/test_sim_full_fossen.py --vehicle sysid --save-plot results/fossen.png
```

This diagnostic plots validation **force and torque**, because those are its
predicted outputs. Its target is requested wrench, not measured delivered
thrust. To evaluate predicted states, pass measured and predicted states to the
same functions. Forward motion prediction still requires a model simulator;
these utilities do not integrate equations of motion.

# Forward Fossen dynamics

`fossen_model.fossen_acceleration` returns the six body-velocity derivatives.
`fossen_model.fossen_dynamics` also integrates position and attitude, with state:

```text
[north, east, depth, qx, qy, qz, qw, u, v, w, p, q, r]
```

The quaternion is scalar-last and rotates body to NED. The 33 fitted parameters
use the same order as `build_regression_fossen`. Bind the known physical
constants once to obtain `f(x, theta, forces)`:

```python
from functools import partial
import numpy as np
from system_identification.src.fossen_model import fossen_dynamics
from system_identification.src.evaluate import trace_dynamics

f = partial(fossen_dynamics,
            mass_kg=11.5992, volume=0.0108320817855168,
            center_of_gravity=[0, 0, 0.006])
x0 = np.array([0, 0, 100, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
forces = np.array([2, 0, 0, 0, 0, 0])
x_dot = f(x0, fitted_parameters, forces)
trajectory = trace_dynamics(x0, f, np.linspace(0, 10, 501), fitted_parameters,
                            input_forces=forces, max_step=0.02)
# trajectory[:, :3]: position; trajectory[:, 7:]: body velocities.
```

This assumes still water and full submersion, with no actuator dynamics.
Forces/torques must be applied at the same body origin used for the state and
inertia. A wrench specified at CG must have its moment shifted to that origin.
The function rejects non-positive-definite inertia; damping signs are left as
fitted. Quaternions are normalized internally when computing attitude; numerical
integration may still leave small norm errors in the returned trajectory.
