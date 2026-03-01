"""Dynamics and cost functions ported from CloseLoopPushing/model/generated/*.m

All functions are direct translations of the Symbolic Math Toolbox output.
D_inv is stored column-major (Fortran order) in MATLAB as a 3x3 matrix.
In Python we use row-major numpy arrays, so indexing is adjusted:
  MATLAB in4(1)->D_inv[0,0], in4(2)->D_inv[1,0], in4(3)->D_inv[2,0]
  MATLAB in4(4)->D_inv[0,1], in4(5)->D_inv[1,1], in4(6)->D_inv[2,1]
"""

import numpy as np


def f_(x, u, T, D_inv):
    """State dynamics: x_{k+1} = f(x_k, u_k).

    Args:
        x: state (3,)  [x1, x2, x3=theta]
        u: control (2,) [u1=angle, u2=magnitude]
        T: timestep
        D_inv: 3x3 inverse contact model matrix
    Returns:
        f: next state (3,)
    """
    d_1_1 = D_inv[0, 0]
    d_1_2 = D_inv[0, 1]
    d_2_1 = D_inv[1, 0]
    d_2_2 = D_inv[1, 1]
    d_3_1 = D_inv[2, 0]
    d_3_2 = D_inv[2, 1]
    u1, u2 = u[0], u[1]
    x1, x2, x3 = x[0], x[1], x[2]

    t2 = np.sin(x3)
    t3 = np.cos(x3)
    t4 = np.cos(u1)
    t5 = np.sin(u1)

    f = np.array([
        x1 + t4 * u2 * (T * d_1_1 * t3 - T * d_2_1 * t2) + t5 * u2 * (T * d_1_2 * t3 - T * d_2_2 * t2),
        x2 + T * t4 * u2 * (d_1_1 * t2 + d_2_1 * t3) + T * t5 * u2 * (d_1_2 * t2 + d_2_2 * t3),
        x3 + T * d_3_1 * t4 * u2 + T * d_3_2 * t5 * u2,
    ])
    return f


def fx_(x, u, T, D_inv):
    """Jacobian of dynamics w.r.t. state: df/dx (3x3)."""
    d_1_1 = D_inv[0, 0]
    d_1_2 = D_inv[0, 1]
    d_2_1 = D_inv[1, 0]
    d_2_2 = D_inv[1, 1]
    u1, u2 = u[0], u[1]
    x3 = x[2]

    t2 = np.sin(x3)
    t3 = np.cos(x3)
    t4 = np.cos(u1)
    t5 = np.sin(u1)

    # MATLAB reshape([...], [3,3]) fills column-major
    # fx(row, col):
    # col0: [1, 0, 0]
    # col1: [0, 1, 0]
    # col2: [-t4*u2*(T*d11*t2+T*d21*t3)-t5*u2*(T*d12*t2+T*d22*t3),
    #         T*t4*u2*(d11*t3-d21*t2)+T*t5*u2*(d12*t3-d22*t2),
    #         1]
    fx = np.array([
        [1.0, 0.0,
         -t4 * u2 * (T * d_1_1 * t2 + T * d_2_1 * t3) - t5 * u2 * (T * d_1_2 * t2 + T * d_2_2 * t3)],
        [0.0, 1.0,
         T * t4 * u2 * (d_1_1 * t3 - d_2_1 * t2) + T * t5 * u2 * (d_1_2 * t3 - d_2_2 * t2)],
        [0.0, 0.0, 1.0],
    ])
    return fx


def fu_(x, u, T, D_inv):
    """Jacobian of dynamics w.r.t. control: df/du (3x2)."""
    d_1_1 = D_inv[0, 0]
    d_1_2 = D_inv[0, 1]
    d_2_1 = D_inv[1, 0]
    d_2_2 = D_inv[1, 1]
    d_3_1 = D_inv[2, 0]
    d_3_2 = D_inv[2, 1]
    u1, u2 = u[0], u[1]
    x3 = x[2]

    t2 = np.sin(x3)
    t3 = np.cos(x3)
    t4 = np.cos(u1)
    t5 = T * d_1_1 * t3
    t6 = np.sin(u1)
    t7 = T * d_1_2 * t3
    t8 = d_2_1 * t3
    t9 = d_1_1 * t2
    t10 = t8 + t9
    t11 = d_2_2 * t3
    t12 = d_1_2 * t2
    t13 = t11 + t12

    # MATLAB reshape([...], [3,2]) fills column-major
    # col0 (du1): row0, row1, row2
    # col1 (du2): row0, row1, row2
    fu = np.array([
        [-t6 * u2 * (t5 - T * d_2_1 * t2) + t4 * u2 * (t7 - T * d_2_2 * t2),
         t4 * (t5 - T * d_2_1 * t2) + t6 * (t7 - T * d_2_2 * t2)],
        [-T * t6 * t10 * u2 + T * t4 * t13 * u2,
         T * t4 * t10 + T * t6 * t13],
        [T * u2 * (d_3_2 * t4 - d_3_1 * t6),
         T * (d_3_1 * t4 + d_3_2 * t6)],
    ])
    return fu


def _safe_acos_cos(dtheta):
    """Compute acos(cos(dtheta)) safely, equivalent to wrapping angle to [0, pi]."""
    c = np.cos(dtheta)
    c = np.clip(c, -1.0, 1.0)
    return np.arccos(c)


def L_(x, u, Q, R, ep, rho, xstar):
    """Running cost (pseudo-Huber loss).

    Args:
        x: state (3,)
        u: control (2,)
        Q: state cost weights (3,)
        R: control cost weights (2,)
        ep: Huber parameters (3,)
        rho: characteristic length
        xstar: reference state (3,)
    Returns:
        scalar cost
    """
    t2 = x[0] - xstar[0]
    t3 = x[1] - xstar[1]
    t4 = _safe_acos_cos(x[2] - xstar[2])

    L = (-Q[0] * (ep[0] - np.sqrt(ep[0]**2 + t2**2))
         - Q[1] * (ep[1] - np.sqrt(ep[1]**2 + t3**2))
         + R[0] * u[0]**2 + R[1] * u[1]**2
         - Q[2] * rho * (ep[2] - np.sqrt(ep[2]**2 + t4**2)))
    return L


def Final_(x, Q, ep, rho, xstar):
    """Terminal cost."""
    t2 = x[0] - xstar[0]
    t3 = x[1] - xstar[1]
    t4 = _safe_acos_cos(x[2] - xstar[2])

    F = (-Q[0] * (ep[0] - np.sqrt(ep[0]**2 + t2**2))
         - Q[1] * (ep[1] - np.sqrt(ep[1]**2 + t3**2))
         - Q[2] * rho * (ep[2] - np.sqrt(ep[2]**2 + t4**2)))
    return F


def Finalx_(x, Q, ep, rho, xstar):
    """Gradient of terminal cost w.r.t. state (3,)."""
    t2 = x[0] - xstar[0]
    t3 = x[1] - xstar[1]
    t4 = x[2] - xstar[2]
    t5 = np.cos(t4)
    t5c = np.clip(t5, -1.0, 1.0)
    t6 = np.arccos(t5c)

    sin_t4 = np.sin(t4)
    denom3 = np.sqrt(np.clip(1.0 - t5c**2, 1e-30, None))

    Fx = np.array([
        Q[0] * t2 / np.sqrt(ep[0]**2 + t2**2),
        Q[1] * t3 / np.sqrt(ep[1]**2 + t3**2),
        Q[2] * rho * t6 * sin_t4 / denom3 / np.sqrt(ep[2]**2 + t6**2),
    ])
    return Fx


def Finalxx_(x, Q, ep, rho, xstar):
    """Hessian of terminal cost w.r.t. state (3x3), diagonal."""
    t2 = x[0] - xstar[0]
    t3 = ep[0]**2
    t4 = x[1] - xstar[1]
    t5 = ep[1]**2
    t6 = ep[2]**2
    t7 = _safe_acos_cos(x[2] - xstar[2])

    Fxx = np.diag([
        Q[0] * t3 / (t3 + t2**2)**1.5,
        Q[1] * t5 / (t5 + t4**2)**1.5,
        Q[2] * rho * t6 / (t6 + t7**2)**1.5,
    ])
    return Fxx


def Hx_(x, u, T, D_inv, Q, R, ep, rho, xstar, lbd):
    """Gradient of Hamiltonian w.r.t. state (3,).

    lbd: costate / lambda (3,)
    """
    d_1_1 = D_inv[0, 0]
    d_1_2 = D_inv[0, 1]
    d_2_1 = D_inv[1, 0]
    d_2_2 = D_inv[1, 1]
    u1, u2 = u[0], u[1]
    x1, x2, x3 = x[0], x[1], x[2]
    xs1, xs2, xs3 = xstar[0], xstar[1], xstar[2]
    lbd1, lbd2, lbd3 = lbd[0], lbd[1], lbd[2]

    t2 = x1 - xs1
    t3 = x2 - xs2
    t4 = np.cos(x3)
    t5 = np.sin(x3)
    t6 = np.cos(u1)
    t7 = np.sin(u1)
    t8 = x3 - xs3
    t9 = np.cos(t8)
    t9c = np.clip(t9, -1.0, 1.0)
    t10 = np.arccos(t9c)

    sin_t8 = np.sin(t8)
    denom = np.sqrt(np.clip(1.0 - t9c**2, 1e-30, None))

    Hx = np.array([
        lbd1 + Q[0] * (x1 * 2.0 - xs1 * 2.0) / (2.0 * np.sqrt(ep[0]**2 + t2**2)),
        lbd2 + Q[1] * (x2 * 2.0 - xs2 * 2.0) / (2.0 * np.sqrt(ep[1]**2 + t3**2)),
        (lbd3
         - lbd1 * (T * t6 * u2 * (d_1_1 * t5 + d_2_1 * t4) + T * t7 * u2 * (d_1_2 * t5 + d_2_2 * t4))
         + lbd2 * (T * t6 * u2 * (d_1_1 * t4 - d_2_1 * t5) + T * t7 * u2 * (d_1_2 * t4 - d_2_2 * t5))
         + Q[2] * rho * t10 * sin_t8 / denom / np.sqrt(ep[2]**2 + t10**2)),
    ])
    return Hx


def Hu_(x, u, T, D_inv, Q, R, ep, rho, xstar, lbd):
    """Gradient of Hamiltonian w.r.t. control (2,)."""
    d_1_1 = D_inv[0, 0]
    d_1_2 = D_inv[0, 1]
    d_2_1 = D_inv[1, 0]
    d_2_2 = D_inv[1, 1]
    d_3_1 = D_inv[2, 0]
    d_3_2 = D_inv[2, 1]
    u1, u2 = u[0], u[1]
    x3 = x[2]
    lbd1, lbd2, lbd3 = lbd[0], lbd[1], lbd[2]

    t2 = np.cos(x3)
    t3 = np.sin(x3)
    t4 = np.cos(u1)
    t5 = np.sin(u1)
    t6 = d_2_1 * t2 + d_1_1 * t3   # t8 in MATLAB
    t7 = d_2_2 * t2 + d_1_2 * t3   # t11
    t8 = d_1_1 * t2 - d_2_1 * t3   # t13
    t9 = d_1_2 * t2 - d_2_2 * t3   # t15

    Hu = np.array([
        (R[0] * u1 * 2.0
         - lbd2 * (T * t5 * t6 * u2 - T * t4 * t7 * u2)
         - lbd1 * (T * t5 * t8 * u2 - T * t4 * t9 * u2)
         - T * lbd3 * u2 * (d_3_1 * t5 - d_3_2 * t4)),
        (R[1] * u2 * 2.0
         + lbd2 * (T * t4 * t6 + T * t5 * t7)
         + lbd1 * (T * t4 * t8 + T * t5 * t9)
         + T * lbd3 * (d_3_1 * t4 + d_3_2 * t5)),
    ])
    return Hu


def Hxx_(x, u, T, D_inv, Q, R, ep, rho, xstar, lbd):
    """Hessian of Hamiltonian w.r.t. state (3x3)."""
    d_1_1 = D_inv[0, 0]
    d_1_2 = D_inv[0, 1]
    d_2_1 = D_inv[1, 0]
    d_2_2 = D_inv[1, 1]
    u1, u2 = u[0], u[1]
    x1, x2, x3 = x[0], x[1], x[2]
    xs1, xs2, xs3 = xstar[0], xstar[1], xstar[2]
    lbd1, lbd2 = lbd[0], lbd[1]

    t2 = x1 - xs1
    t3 = ep[0]**2
    t4 = x2 - xs2
    t5 = ep[1]**2
    t6 = np.cos(x3)
    t7 = np.sin(x3)
    t8 = np.cos(u1)
    t9 = np.sin(u1)
    t11 = x3 - xs3
    t12 = np.cos(t11)
    t12c = np.clip(t12, -1.0, 1.0)
    t10 = np.arccos(t12c)
    t13 = t10**2
    t14 = ep[2]**2
    t15 = t13 + t14

    val_33 = (
        - lbd2 * (T * t8 * u2 * (d_1_1 * t7 + d_2_1 * t6) + T * t9 * u2 * (d_1_2 * t7 + d_2_2 * t6))
        - lbd1 * (T * t8 * u2 * (d_1_1 * t6 - d_2_1 * t7) + T * t9 * u2 * (d_1_2 * t6 - d_2_2 * t7))
        + Q[2] * rho / np.sqrt(t15)
        - Q[2] * rho * t13 / t15**1.5
    )

    Hxx = np.array([
        [Q[0] * t3 / (t3 + t2**2)**1.5, 0.0, 0.0],
        [0.0, Q[1] * t5 / (t5 + t4**2)**1.5, 0.0],
        [0.0, 0.0, val_33],
    ])
    return Hxx


def Huu_(x, u, T, D_inv, Q, R, ep, rho, xstar, lbd):
    """Hessian of Hamiltonian w.r.t. control (2x2)."""
    d_1_1 = D_inv[0, 0]
    d_1_2 = D_inv[0, 1]
    d_2_1 = D_inv[1, 0]
    d_2_2 = D_inv[1, 1]
    d_3_1 = D_inv[2, 0]
    d_3_2 = D_inv[2, 1]
    u1, u2 = u[0], u[1]
    x3 = x[2]
    lbd1, lbd2, lbd3 = lbd[0], lbd[1], lbd[2]

    t2 = np.cos(x3)
    t3 = np.sin(x3)
    t4 = np.cos(u1)
    t5 = np.sin(u1)
    t6 = d_2_2 * t2 + d_1_2 * t3   # t8
    t8 = d_2_1 * t2 + d_1_1 * t3   # t11
    t12 = d_1_2 * t2 - d_2_2 * t3  # t13
    t14 = d_1_1 * t2 - d_2_1 * t3  # t15

    t16 = T * t4 * t6
    t17 = t16 - T * t5 * t8
    t18 = lbd2 * t17

    t20 = T * t4 * t12
    t22 = t20 - T * t5 * t14
    t23 = lbd1 * t22

    t24 = d_3_1 * t5
    t25 = t24 - d_3_2 * t4
    t26 = t18 + t23 - T * lbd3 * t25

    val_00 = (R[0] * 2.0
              - lbd2 * (T * t5 * t6 * u2 + T * t4 * t8 * u2)
              - lbd1 * (T * t5 * t12 * u2 + T * t4 * t14 * u2)
              - T * lbd3 * u2 * (d_3_1 * t4 + d_3_2 * t5))

    Huu = np.array([
        [val_00, t26],
        [t26, R[1] * 2.0],
    ])
    return Huu


def Hux_(x, u, T, D_inv, Q, R, ep, rho, xstar, lbd):
    """Mixed Hessian of Hamiltonian d²H/(du dx) (2x3)."""
    d_1_1 = D_inv[0, 0]
    d_1_2 = D_inv[0, 1]
    d_2_1 = D_inv[1, 0]
    d_2_2 = D_inv[1, 1]
    u1, u2 = u[0], u[1]
    x3 = x[2]
    lbd1, lbd2 = lbd[0], lbd[1]

    t2 = np.cos(x3)
    t3 = np.sin(x3)
    t4 = np.cos(u1)
    t5 = np.sin(u1)
    t6 = d_2_1 * t2 + d_1_1 * t3   # t8
    t7 = d_2_2 * t2 + d_1_2 * t3   # t11
    t8 = d_1_1 * t2 - d_2_1 * t3   # t13
    t9 = d_1_2 * t2 - d_2_2 * t3   # t15

    # MATLAB reshape([0,0,0,0, val1, val2], [2,3]) column-major
    # row0 = [0, 0, val1]
    # row1 = [0, 0, val2]
    Hux = np.array([
        [0.0, 0.0,
         lbd1 * (T * t5 * t6 * u2 - T * t4 * t7 * u2) - lbd2 * (T * t5 * t8 * u2 - T * t4 * t9 * u2)],
        [0.0, 0.0,
         -lbd1 * (T * t4 * t6 + T * t5 * t7) + lbd2 * (T * t4 * t8 + T * t5 * t9)],
    ])
    return Hux
