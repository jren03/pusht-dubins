"""DDP/iLQR solver ported from DDP/DDP_codeGen.m and DDP/DDP_getDefaultPara.m.

Solves:
    minimize  sum_i L(x_i, u_i) + Final(x_N)
       u
    s.t.  x_{i+1} = f(x_i, u_i)
          uLB <= u <= uUB

Reference:
    Jacobson D, Mayne D. Differential dynamic programming. 1970.
    Tassa Y, Mansard N, Todorov E. Control-limited differential dynamic programming. ICRA 2014.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from . import model as mdl


@dataclass
class DDPPara:
    detail: int = 1
    Ci_reg_init: float = 1.0
    Ci_reg_Min: float = 1e-7
    Ci_reg_Max: float = 1e8
    Ci_reg_fac: float = 1.6
    Ci_reg_localloop: int = 3
    maxIter: int = 300
    tolGrad: float = 1e-4
    MinCostChange: float = 1e-9
    maxInnerLoop: int = 10
    c: float = 0.5


@dataclass
class DDPInfo:
    exitflag: int = 0
    exitinfo: str = "info"
    iter: int = 1


def get_default_para() -> DDPPara:
    return DDPPara()


def _wrap_angle(theta):
    """Wrap angle to [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def _truncate(x, lb, ub):
    return np.clip(x, lb, ub)


# ---------------------------------------------------------------
#  Wrapped dynamics / cost functions (matching DDP_codeGen.m local functions)
# ---------------------------------------------------------------

def _f(opts, x, u):
    """Forward dynamics with angle wrapping."""
    xnxt = mdl.f_(x, u, opts.T, opts.D_inv)
    xnxt[2] = _wrap_angle(xnxt[2])
    return xnxt


def _f_grad(opts, x, u):
    """Forward dynamics + Jacobians."""
    xnxt = _f(opts, x, u)
    fx = mdl.fx_(x, u, opts.T, opts.D_inv)
    fu = mdl.fu_(x, u, opts.T, opts.D_inv)
    return xnxt, fx, fu


def _L(opts, x, u, i):
    """Running cost at timestep i."""
    return mdl.L_(x, u, opts.Q, opts.R, opts.ep, opts.rho, opts.xref[:, i])


def _Final(opts, x):
    """Terminal cost (scalar only)."""
    return mdl.Final_(x, opts.Qf, opts.epf, opts.rho, opts.xref[:, -1])


def _Final_grad(opts, x):
    """Terminal cost + gradient + Hessian."""
    F = mdl.Final_(x, opts.Qf, opts.epf, opts.rho, opts.xref[:, -1])

    # Angle wrapping + singularity avoidance
    xw = x.copy()
    xw[2] = _wrap_angle(xw[2] - opts.xref[2, -1]) + opts.xref[2, -1]
    if abs(np.sin(xw[2] - opts.xref[2, -1])) < 1e-6:
        xw[2] += 1e-6

    Fx = mdl.Finalx_(xw, opts.Qf, opts.epf, opts.rho, opts.xref[:, -1])
    Fxx = mdl.Finalxx_(xw, opts.Qf, opts.epf, opts.rho, opts.xref[:, -1])
    return F, Fx, Fxx


def _H_grad(opts, x, u, lbd, i):
    """Hamiltonian derivatives with angle wrapping."""
    xw = x.copy()
    xw[2] = _wrap_angle(xw[2] - opts.xref[2, i]) + opts.xref[2, i]
    if abs(np.sin(xw[2] - opts.xref[2, i])) < 1e-6:
        xw[2] += 1e-6

    xstar_i = opts.xref[:, i]
    Hx = mdl.Hx_(xw, u, opts.T, opts.D_inv, opts.Q, opts.R, opts.ep, opts.rho, xstar_i, lbd)
    Hxx = mdl.Hxx_(xw, u, opts.T, opts.D_inv, opts.Q, opts.R, opts.ep, opts.rho, xstar_i, lbd)
    Hu = mdl.Hu_(xw, u, opts.T, opts.D_inv, opts.Q, opts.R, opts.ep, opts.rho, xstar_i, lbd)
    Hux = mdl.Hux_(xw, u, opts.T, opts.D_inv, opts.Q, opts.R, opts.ep, opts.rho, xstar_i, lbd)
    Huu = mdl.Huu_(xw, u, opts.T, opts.D_inv, opts.Q, opts.R, opts.ep, opts.rho, xstar_i, lbd)
    return Hx, Hxx, Hu, Hux, Huu


# ---------------------------------------------------------------
#  boxQP solver (ported from DDP_codeGen.m)
# ---------------------------------------------------------------

def box_qp(H, g, lower, upper, x0=None):
    """Minimize 0.5*x'*H*x + x'*g  s.t. lower <= x <= upper.

    Returns: (x, result, Hfree_chol, free_mask)
    """
    maxIter = 100
    minGrad = 1e-8
    minRelImprove = 1e-8
    stepDec = 0.6
    minStep = 1e-22
    Armijo = 0.1

    n = H.shape[0]
    clamped = np.zeros(n, dtype=bool)
    free = np.ones(n, dtype=bool)
    result = 0
    Hfree = np.zeros((n, n))

    # Initial state
    if x0 is not None and x0.size == n:
        x = np.clip(x0.ravel(), lower, upper)
    else:
        LU = np.column_stack([lower, upper]).astype(float)
        LU[~np.isfinite(LU)] = np.nan
        x = np.nanmean(LU, axis=1)
    x[~np.isfinite(x)] = 0.0

    value = x @ g + 0.5 * x @ H @ x
    oldvalue = 0.0

    for it in range(1, maxIter + 1):
        if result != 0:
            break

        # Check relative improvement
        if it > 1 and (oldvalue - value) < minRelImprove * abs(oldvalue):
            result = 4
            break
        oldvalue = value

        # Gradient
        grad = g + H @ x

        # Find clamped dimensions
        old_clamped = clamped.copy()
        clamped = np.zeros(n, dtype=bool)
        clamped[(x == lower) & (grad > 0)] = True
        clamped[(x == upper) & (grad < 0)] = True
        free = ~clamped

        if np.all(clamped):
            result = 6
            break

        # Factorize if clamped changed
        if it == 1 or np.any(old_clamped != clamped):
            try:
                Hfree = np.linalg.cholesky(H[np.ix_(free, free)])
            except np.linalg.LinAlgError:
                result = -1
                break

        # Check gradient norm
        gnorm = np.linalg.norm(grad[free])
        if gnorm < minGrad:
            result = 5
            break

        # Search direction
        grad_clamped = g + H @ (x * clamped)
        search = np.zeros(n)
        # search[free] = -Hfree\(Hfree'\grad_clamped[free]) - x[free]
        z = np.linalg.solve(Hfree, grad_clamped[free])
        z = np.linalg.solve(Hfree.T, z)
        search[free] = -z - x[free]

        # Check descent
        sdotg = search @ grad
        if sdotg >= 0:
            break

        # Armijo line search
        step = 1.0
        xc = np.clip(x + step * search, lower, upper)
        vc = xc @ g + 0.5 * xc @ H @ xc
        while (vc - oldvalue) / (step * sdotg) < Armijo:
            step *= stepDec
            xc = np.clip(x + step * search, lower, upper)
            vc = xc @ g + 0.5 * xc @ H @ xc
            if step < minStep:
                result = 2
                break

        x = xc
        value = vc

    if it >= maxIter:
        result = 1

    return x, result, Hfree, free


# ---------------------------------------------------------------
#  Main DDP solver
# ---------------------------------------------------------------

def ddp_solve(x0, unom, opts, uLB=None, uUB=None, para=None):
    """DDP/iLQR trajectory optimization.

    Args:
        x0: initial state (nx,)
        unom: initial control sequence (nu, N)
        opts: OptsModel with dynamics/cost parameters
        uLB: optional lower bounds on control (nu, N)
        uUB: optional upper bounds on control (nu, N)
        para: DDPPara solver parameters

    Returns:
        xnom: optimized state trajectory (nx, N)
        unom: optimized control trajectory (nu, N)
        alpha_: feedforward gains (nu, N)
        beta_: feedback gains (nu, nx, N)
        info: DDPInfo
    """
    if para is None:
        para = get_default_para()

    info = DDPInfo()
    print_head = 10
    print_count = print_head

    nx = x0.shape[0]
    nu, N = unom.shape

    Ci_reg_base = para.Ci_reg_init
    Ci_reg_ramp = 1.0

    is_constrained = uLB is not None

    xnom = np.zeros((nx, N))
    alpha_ = np.zeros((nu, N))
    beta_ = np.zeros((nu, nx, N))

    flag_backpass_success = True

    for iteration in range(1, para.maxIter + 1):
        info.iter = iteration

        # -----------------------------------------------
        # Step 1: Forward pass - get nominal trajectory
        # -----------------------------------------------
        xnom = np.zeros((nx, N))
        xnom[:, 0] = x0
        fx_arr = np.zeros((nx, nx, N - 1))
        fu_arr = np.zeros((nx, nu, N - 1))

        Vnom = 0.0
        for i in range(N - 1):
            xnom[:, i + 1], fx_arr[:, :, i], fu_arr[:, :, i] = _f_grad(opts, xnom[:, i], unom[:, i])
            Vnom += _L(opts, xnom[:, i], unom[:, i], i)
        Vnom += _Final(opts, xnom[:, N - 1])

        # -----------------------------------------------
        # Step 2: Backward pass
        # -----------------------------------------------
        a = np.zeros(N)
        Vx = np.zeros((nx, N))
        Vxx = np.zeros((nx, nx, N))
        alpha_ = np.zeros((nu, N))
        beta_ = np.zeros((nu, nx, N))

        flag_backpass_success = True

        # Boundary condition
        _, Vx[:, N - 1], Vxx[:, :, N - 1] = _Final_grad(opts, xnom[:, N - 1])

        Ci_reg = 0.0
        for i in range(N - 2, -1, -1):
            Hx, Hxx_val, Hu, Hux_val, Huu_val = _H_grad(
                opts, xnom[:, i], unom[:, i], Vx[:, i + 1], i
            )

            fxi = fx_arr[:, :, i]
            fui = fu_arr[:, :, i]
            Vxxi1 = Vxx[:, :, i + 1]

            Ai = Hxx_val + fxi.T @ Vxxi1 @ fxi
            Bi = Hux_val + fui.T @ Vxxi1 @ fxi
            Ci = Huu_val + fui.T @ Vxxi1 @ fui

            # Regularization
            reg_success = False
            for r in range(para.Ci_reg_localloop + 1):
                Ci_reg = Ci_reg_base * (2 ** r)
                Ci_r = Ci + Ci_reg * np.eye(nu)
                cond_Ci = np.linalg.cond(Ci_r)
                if cond_Ci < 1.0 / (100 * np.finfo(float).eps):
                    reg_success = True
                    break

            if not reg_success:
                flag_backpass_success = False
                break

            Ci = Ci_r

            if not is_constrained:
                alpha_[:, i] = -np.linalg.solve(Ci, Hu)
                beta_[:, :, i] = -np.linalg.solve(Ci, Bi)
            else:
                # Constrained: boxQP
                x0_qp = alpha_[:, min(i + 1, N - 1)]
                alpha_[:, i], qp_result, R_chol, free_mask = box_qp(
                    Ci, Hu,
                    uLB[:, i] - unom[:, i],
                    uUB[:, i] - unom[:, i],
                    x0_qp,
                )
                if qp_result < 1:
                    flag_backpass_success = False
                    break
                beta_[:, :, i] = np.zeros((nu, nx))
                if np.any(free_mask):
                    # Lfree = -R\(R'\Bi(free,:))
                    Bi_free = Bi[free_mask, :]
                    z = np.linalg.solve(R_chol, Bi_free)
                    Lfree = -np.linalg.solve(R_chol.T, z)
                    beta_[free_mask, :, i] = Lfree

            if is_constrained:
                a[i] = a[i + 1] + Hu @ alpha_[:, i] + 0.5 * alpha_[:, i] @ Ci @ alpha_[:, i]
                Vx[:, i] = Hx + beta_[:, :, i].T @ Hu + (Ci @ beta_[:, :, i] + Bi).T @ alpha_[:, i]
                Vxx[:, :, i] = (Ai + beta_[:, :, i].T @ Ci @ beta_[:, :, i]
                                + beta_[:, :, i].T @ Bi + Bi.T @ beta_[:, :, i])
            else:
                a[i] = a[i + 1] + 0.5 * Hu @ alpha_[:, i]
                Vx[:, i] = Hx + beta_[:, :, i].T @ Hu
                Vxx[:, :, i] = Ai - beta_[:, :, i].T @ Ci @ beta_[:, :, i]

            Vxx[:, :, i] = 0.5 * (Vxx[:, :, i] + Vxx[:, :, i].T)

        # -----------------------------------------------
        # Step 3: Check gradient
        # -----------------------------------------------
        flag_linesearch_success = False
        deltaV = 0.0
        a0 = 0.0
        g_norm = 0.0
        V = 0.0

        if flag_backpass_success:
            g_norm = np.mean(np.max(np.abs(alpha_) / (np.abs(unom) + 1), axis=0))
            if g_norm < para.tolGrad and Ci_reg < 1e-5:
                info.exitflag = 1
                info.exitinfo = "small gradient"
                if para.detail >= 1:
                    print("Terminated with success: small gradient")
                break

            # -----------------------------------------------
            # Step 4: Line search
            # -----------------------------------------------
            x_trial = np.zeros((nx, N))
            x_trial[:, 0] = x0
            u_trial = np.zeros((nu, N))

            inner_count = 1
            epsilon = 1.0
            a0 = a[0]

            while True:
                # Forward pass with new policy
                for i in range(N - 1):
                    deltaX = x_trial[:, i] - xnom[:, i]
                    u_trial[:, i] = unom[:, i] + epsilon * alpha_[:, i] + beta_[:, :, i] @ deltaX
                    if is_constrained:
                        u_trial[0, i] = _truncate(u_trial[0, i], uLB[0, i], uUB[0, i])
                        u_trial[1, i] = _truncate(u_trial[1, i], uLB[1, i], uUB[1, i])
                    x_trial[:, i + 1] = _f(opts, x_trial[:, i], u_trial[:, i])

                # Evaluate cost
                V = 0.0
                for i in range(N - 1):
                    V += _L(opts, x_trial[:, i], u_trial[:, i], i)
                V += _Final(opts, x_trial[:, N - 1])

                deltaV = V - Vnom

                if a0 < 0:
                    V_improvement = deltaV / a0
                else:
                    V_improvement = -np.sign(deltaV)
                    print("[WARNING] Expected change in Value is positive")

                if V_improvement > para.c:
                    flag_linesearch_success = True
                    break

                epsilon *= 0.5
                a0 = 2 * epsilon * (1 - epsilon / 2) * a[0]

                inner_count += 1
                if inner_count > para.maxInnerLoop:
                    break

        # -----------------------------------------------
        # Step 5: Update
        # -----------------------------------------------
        if para.detail >= 1 and print_count == print_head:
            print_count = 0
            print(f"{'Iter':<12}{'Value':<12}{'reduction':<12}{'expected':<12}{'gradient':<12}{'log10(reg)':<12}")

        if flag_linesearch_success:
            unom = u_trial.copy()

            if para.detail >= 1:
                print(f"{iteration:<12d}{V:<12.6g}{deltaV:<12.3g}{a0:<12.3g}{g_norm:<12.3g}{np.log10(max(Ci_reg_base, 1e-30)):<12.1f}")
                print_count += 1

            # Shrink regularization
            Ci_reg_ramp = min(Ci_reg_ramp / para.Ci_reg_fac, 1.0 / para.Ci_reg_fac)
            if Ci_reg_base > para.Ci_reg_Min:
                Ci_reg_base = Ci_reg_base * Ci_reg_ramp
            else:
                Ci_reg_base = 0.0

            if abs(deltaV) < para.MinCostChange:
                info.exitflag = 2
                info.exitinfo = "small cost change"
                if para.detail >= 1:
                    print("Terminated with success: change in cost very small")
                break
        else:
            if para.detail >= 1:
                if flag_backpass_success:
                    print(f"{iteration:<12d}{'LS-FAIL':<12}{deltaV:<12.3g}{a0:<12.3g}{g_norm:<12.3g}{np.log10(max(Ci_reg_base, 1e-30)):<12.1f}")
                else:
                    print(f"{iteration:<12d}{'BP-FAIL':<12}{'-':<12}{'-':<12}{'-':<12}{np.log10(max(Ci_reg_base, 1e-30)):<12.1f}")
                print_count += 1

            Ci_reg_ramp = max(Ci_reg_ramp * para.Ci_reg_fac, para.Ci_reg_fac)
            Ci_reg_base = max(Ci_reg_base * Ci_reg_ramp, para.Ci_reg_Min)

            if Ci_reg_base > para.Ci_reg_Max:
                info.exitflag = -1
                info.exitinfo = "ill-conditioned"
                if para.detail >= 1:
                    print("Terminated with failure: ill conditioned")
                break

    if not flag_backpass_success:
        info.exitflag = -1

    if para.detail > 0:
        print(f"DDP terminated with number of iterations: {info.iter}")

    # Recompute final xnom with the final unom
    xnom[:, 0] = x0
    for i in range(N - 1):
        xnom[:, i + 1] = _f(opts, xnom[:, i], unom[:, i])

    return xnom, unom, alpha_, beta_, info
