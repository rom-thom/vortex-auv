# Fossen regression

`build_regression_fossen` uses the six-DOF relative-motion equation

```text
M nu_r_dot + C(M, nu_r) nu_r + Dl nu_r + Dq (abs(nu_r) * nu_r) + g = tau
```

It is nonlinear in velocity but linear in the estimated parameters. This
implementation estimates a constant symmetric total inertia matrix (21 entries)
and diagonal linear/quadratic damping (12 entries). Coriolis terms use the same
inertia entries; they are not independently fitted. Diagonal damping is a
model simplification: cross-axis damping, lift and other loads are not fitted.
Known hydrostatic restoring forces/moments are subtracted from the target.

## Usage

```python
from system_identification.src.regresion_builds import (
    FOSSEN_PARAMETER_NAMES, build_regression_fossen,
)
from system_identification.src.system_identification import SystemIdentification

# Each array has shape (N, 6), synchronized and in the same body frame/origin.
data = {
    "velocity": relative_velocity,         # [u, v, w, p, q, r]
    "acceleration": relative_velocity_dot,
    "force": applied_wrench,               # [X, Y, Z, K, M, N]
    "restoring": hydrostatic_wrench,       # g on the LEFT side of the equation
}
model = SystemIdentification(regression_builder=build_regression_fossen)
training, validation = model.split_data(data, 0.7)
parameters = model.fit_parameters(training)
print(dict(zip(FOSSEN_PARAMETER_NAMES, parameters)))

# predict() reconstructs the regression target tau - g, flattened sample-first.
predicted_wrench = model.predict(validation).reshape(-1, 6) + validation["restoring"]
rmse_per_axis = ((predicted_wrench - validation["force"]) ** 2).mean(axis=0) ** 0.5
print("RMSE [N, N, N, N*m, N*m, N*m]:", rmse_per_axis)
```

The default `SystemIdentification()` still uses the existing one-dimensional
builder. With Fossen, `score()` combines forces and torques into one number;
prefer the per-axis errors above. Ordinary least squares also weights these
equations equally numerically; uncertainty-based row weighting may be needed
when sensors have different noise levels.

## Preparing measurements

- Velocities are water-relative: `nu_r = nu - nu_current_body`. The simplified
  total-inertia equation assumes no current or a steady, spatially uniform,
  irrotational current. Rotate a known world current into the body frame at
  every timestamp; it is not generally constant in that frame.
- Acceleration means the derivative of the body-frame velocity components,
  not raw IMU acceleration or just gravity-corrected IMU acceleration.
  At the body origin, `v_dot_body = a_body - omega cross v_body`;
  subtract `v_current_body_dot` as well to obtain the relative derivative.
  An offset IMU also needs lever-arm correction. Angular entries are
  `[p_dot, q_dot, r_dot]`, not angular velocity.
- `restoring` is Fossen's left-hand-side `g(eta)`, the negative of the external
  weight-plus-buoyancy wrench. It must come from known weight, buoyancy,
  orientation and their application points. Neutral buoyancy alone does not
  remove restoring moments when the centres of gravity and buoyancy differ.
- Use applied/estimated thrust at the same body origin. A requested wrench can
  differ from delivered thrust due to allocation, saturation and dynamics.

## Interpreting parameters

`FOSSEN_PARAMETER_NAMES` gives the exact column order: `M_11, M_12, ...,
M_16, M_22, ..., M_66`, followed by damping in `[u, v, w, p, q, r]` order.
Off-diagonal inertia entries populate both symmetric positions. These are
total inertia entries, not separately identifiable rigid-body and added masses.
The damping convention is positive resistance on the left-hand side, opposite
to the signed hydrodynamic derivatives often used in Fossen's notation.

Six-DOF excitation is needed to identify all 33 parameters. A surge-only
recording cannot do this; use the 1D builder for that experiment. Check rank
and conditioning of the regression, validate on independent data, and check
that the reconstructed mass matrix is positive definite and damping is
nonnegative. The unconstrained least-squares solver does not enforce these
physical constraints or resolve errors-in-variables bias.

## Recorded simulator data

For a still-water fit to local simulator recordings, run:

```bash
python test/test_sim_full_fossen.py
```

This uses `odom_test.csv` and `wrench_input_test.csv` at native odometry
timestamps, with a chronological 70/30 split. Derivatives are calculated within
each split; edges and samples shallower than 0.7 m are excluded. There is no
current compensation. Record new data after setting the simulator current to
zero; existing recordings retain their original conditions.

The editable constants at the top of the test describe the Nautilus base hull
in `nautilus.scn`: mass 53.7 kg, CG (0, 0, 0.01), CB approximately (0, 0, 0),
and collision-mesh volume 0.0501485267848 m³ at water density 1031 kg/m³.
They approximate fully submerged restoring forces and omit tiny gripper masses.
The test assumes body-frame twist and body-to-NED pose, with force/torque
referenced to the same body origin. It uses requested wrench, not measured
thrust, and ignores actuator lag and saturation. It reports per-axis errors,
constant-training-mean baselines, R² (NaN for constant validation targets),
inertia eigenvalues and damping signs. Passing pytest confirms pipeline
execution, not physical accuracy. Synthetic correctness tests remain in
`test_linear_full_fossen.py` and `test_fossen_regression.py`.

## References

- [Fossen's marine craft model](https://www.fossen.biz/html/marineCraftModel.html):
  inertia, Coriolis structure and relative-motion assumptions.
- [Identification of underwater vehicle hydrodynamic coefficients using free decay tests](https://torarnj.folk.ntnu.no/main_mkVII.pdf):
  structured regressors and least-squares identification.
- [Marine-vehicle modelling, DTU thesis](https://backend.orbit.dtu.dk/ws/files/161971881/mcn_phd_thesis_final_prepress.pdf):
  section 2, diagonal damping approximation and inertia-based Coriolis matrix.
