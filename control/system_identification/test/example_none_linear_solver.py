"""Nonlinear parameter estimation using offline NLS and an augmented EKF.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def state_step(x, u, theta, dt):
    """One forward-Euler step of the nonlinear dynamical system."""
    x1, x2 = x
    th1, th2 = theta

    f1 = -0.4 * x1 + np.sin(th1 * x2) + 0.3 * u[0]
    f2 = -0.2 * x2 + np.cos(th2 * x1) + 0.3 * u[1]

    return x + dt * np.array([f1, f2])


def nls_objective(theta, u, y, dt):
    """Sum of squared simulation errors over the whole measurement sequence."""
    n = y.shape[1]
    x_pred = np.zeros((2, n))

    # Use the same measured initial state as the plotted NLS trajectory.
    x_pred[:, 0] = y[:, 0]

    for k in range(n - 1):
        x_pred[:, k + 1] = state_step(x_pred[:, k], u[:, k], theta, dt)

    residual = y[:, 1:] - x_pred[:, 1:]
    cost = np.sum(residual**2)
    return float(cost) if np.isfinite(cost) else 1e100


def augmented_jacobian(x, u, theta, dt):
    """Jacobian of z[k+1] = [state_step(x[k], u[k], theta, dt); theta]."""
    x1, x2 = x
    th1, th2 = theta

    # Correct derivatives for f2 = -0.2*x2 + cos(th2*x1) + 0.3*u2.
    df_dx = np.array([
        [-0.4, th1 * np.cos(th1 * x2)],
        [-th2 * np.sin(th2 * x1), -0.2],
    ])
    df_dtheta = np.array([
        [x2 * np.cos(th1 * x2), 0.0],
        [0.0, -x1 * np.sin(th2 * x1)],
    ])

    F = np.eye(4)
    F[:2, :2] += dt * df_dx
    F[:2, 2:] = dt * df_dtheta
    return F


def main():
    rng = np.random.default_rng(7)

    dt = 0.02
    T = 20.0
    t = np.arange(0.0, T + dt / 2, dt)
    n = t.size

    theta_true = np.array([1.30, 0.70])

    u = np.zeros((2, n - 1))
    u[0, :] = 0.7 * np.sin(0.7 * t[:-1]) + 0.25 * np.sin(2.1 * t[:-1])
    u[1, :] = 0.6 * np.cos(0.5 * t[:-1]) + 0.20 * np.sin(1.7 * t[:-1])

    x_true = np.zeros((2, n))
    x_true[:, 0] = [0.8, -0.4]

    for k in range(n - 1):
        x_true[:, k + 1] = state_step(x_true[:, k], u[:, k], theta_true, dt)

    sigma_meas = 0.1
    R_meas = sigma_meas**2 * np.eye(2)
    y = x_true + sigma_meas * rng.standard_normal(x_true.shape)

    # ------------------------------------------------------------------
    # Offline nonlinear least squares (NLS)
    # ------------------------------------------------------------------
    # Nelder-Mead can start at zero because it explores nonzero
    # trial parameters even when the derivative vanishes there.
    theta0 = np.array([0.0, 0.0])
    result = minimize(
        nls_objective,
        theta0,
        args=(u, y, dt),
        method="Nelder-Mead",  # MATLAB's fminsearch
        options={"maxiter": 3000, "maxfev": 10000,
                 "xatol": 1e-9, "fatol": 1e-9},
    )
    theta_nls = result.x
    if not result.success:
        print(f"NLS optimizer warning: {result.message}")

    x_nls = np.zeros((2, n))
    x_nls[:, 0] = y[:, 0]
    for k in range(n - 1):
        x_nls[:, k + 1] = state_step(x_nls[:, k], u[:, k], theta_nls, dt)

    # ------------------------------------------------------------------
    # Online augmented extended Kalman filter (EKF)
    # z = [x1, x2, theta1, theta2]
    # ------------------------------------------------------------------
    z_hat = np.zeros((4, n))
    P_store = np.zeros((4, 4, n))

    # Use the first available measurement to initialize the state,
    # and a nonzero initial estimate for theta2.
    z_hat[:, 0] = [y[0, 0], y[1, 0], 0.0, 0.5]
    P = np.diag([0.08**2, 0.08**2, 0.5**2, 0.4**2])
    P_store[:, :, 0] = P

    Q_x = np.diag([1e-5, 1e-5])
    Q_theta = np.diag([2e-6, 2e-6])
    Q_aug = np.zeros((4, 4))
    Q_aug[:2, :2] = Q_x
    Q_aug[2:, 2:] = Q_theta

    H = np.hstack((np.eye(2), np.zeros((2, 2))))
    I4 = np.eye(4)

    for k in range(n - 1):
        z = z_hat[:, k]
        x_k = z[:2]
        theta_k = z[2:]
        u_k = u[:, k]

        x_pred = state_step(x_k, u_k, theta_k, dt)
        z_pred = np.concatenate((x_pred, theta_k))
        F = augmented_jacobian(x_k, u_k, theta_k, dt)
        P_pred = F @ P @ F.T + Q_aug

        innovation = y[:, k + 1] - H @ z_pred
        S = H @ P_pred @ H.T + R_meas
        K = np.linalg.solve(S, H @ P_pred).T

        z_hat[:, k + 1] = z_pred + K @ innovation

        # Joseph covariance update (numerically more robust).
        I_KH = I4 - K @ H
        P = I_KH @ P_pred @ I_KH.T + K @ R_meas @ K.T
        P_store[:, :, k + 1] = P

    theta_ekf = z_hat[2:, :]

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print(f"\nTrue parameters:       theta1 = {theta_true[0]:.4f}, "
          f"theta2 = {theta_true[1]:.4f}")
    print(f"Offline NLS estimate:  theta1 = {theta_nls[0]:.4f}, "
          f"theta2 = {theta_nls[1]:.4f}")
    print(f"Final online EKF:      theta1 = {theta_ekf[0, -1]:.4f}, "
          f"theta2 = {theta_ekf[1, -1]:.4f}")

    # State estimates
    fig1, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    fig1.suptitle("State trajectories")
    for i, ax in enumerate(axes):
        ax.plot(t, x_true[i], "k-", linewidth=1.8, label="True state")
        ax.plot(t, y[i], color="0.7", linewidth=0.7, label="Noisy measurement")
        ax.plot(t, x_nls[i], "b--", linewidth=1.4, label="Offline NLS simulation")
        ax.plot(t, z_hat[i], "r-.", linewidth=1.4, label="Online augmented EKF")
        ax.set_ylabel(fr"$x_{i+1}$")
        ax.grid(True)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig1.tight_layout()

    # Parameter estimates
    fig2, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    fig2.suptitle("Nonlinear parameter estimation")
    for j, ax in enumerate(axes):
        ax.plot(t, np.full_like(t, theta_true[j]), "k-", linewidth=1.8,
                label="True parameter")
        ax.plot(t, theta_ekf[j], "r-", linewidth=1.4, label="Online EKF")
        ax.axhline(theta_nls[j], color="b", linestyle="--", linewidth=1.4,
                   label="Offline NLS estimate")
        std_theta = np.sqrt(np.maximum(P_store[j + 2, j + 2, :], 0.0))
        ax.plot(t, theta_ekf[j] + 2 * std_theta, "r--", linewidth=0.9,
                label=r"EKF $\pm 2\sigma$" if j == 0 else None)
        ax.plot(t, theta_ekf[j] - 2 * std_theta, "r--", linewidth=0.9)
        ax.set_ylabel(fr"$\theta_{j+1}$")
        ax.grid(True)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
