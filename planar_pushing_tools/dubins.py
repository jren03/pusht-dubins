"""Pure Python Dubins curves solver.

Ported from Andrew Walker's C implementation:
  https://github.com/AndrewWalker/Dubins-Curves

A Dubins path connects two (x, y, theta) configurations with the
shortest path composed of circular arcs (L=left, R=right) and straight
segments (S), with a minimum turning radius rho.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# Path type indices
LSL, LSR, RSL, RSR, RLR, LRL = range(6)

# Segment types
L_SEG, S_SEG, R_SEG = 0, 1, 2

DIRDATA = [
    [L_SEG, S_SEG, L_SEG],  # LSL
    [L_SEG, S_SEG, R_SEG],  # LSR
    [R_SEG, S_SEG, L_SEG],  # RSL
    [R_SEG, S_SEG, R_SEG],  # RSR
    [R_SEG, L_SEG, R_SEG],  # RLR
    [L_SEG, R_SEG, L_SEG],  # LRL
]

EPSILON = 1e-10


def _mod2pi(theta):
    return theta - 2.0 * np.pi * np.floor(theta / (2.0 * np.pi))


@dataclass
class DubinsPath:
    qi: np.ndarray      # initial configuration (3,)
    param: np.ndarray   # segment lengths (3,)
    rho: float          # turning radius
    type: int           # path type (0-5)


def _dubins_LSL(alpha, beta, d):
    sa, sb, ca, cb = np.sin(alpha), np.sin(beta), np.cos(alpha), np.cos(beta)
    c_ab = np.cos(alpha - beta)
    tmp0 = d + sa - sb
    p_sq = 2 + d * d - 2 * c_ab + 2 * d * (sa - sb)
    if p_sq < 0:
        return None
    tmp1 = np.arctan2(cb - ca, tmp0)
    t = _mod2pi(-alpha + tmp1)
    p = np.sqrt(p_sq)
    q = _mod2pi(beta - tmp1)
    return np.array([t, p, q])


def _dubins_RSR(alpha, beta, d):
    sa, sb, ca, cb = np.sin(alpha), np.sin(beta), np.cos(alpha), np.cos(beta)
    c_ab = np.cos(alpha - beta)
    tmp0 = d - sa + sb
    p_sq = 2 + d * d - 2 * c_ab + 2 * d * (sb - sa)
    if p_sq < 0:
        return None
    tmp1 = np.arctan2(ca - cb, tmp0)
    t = _mod2pi(alpha - tmp1)
    p = np.sqrt(p_sq)
    q = _mod2pi(-beta + tmp1)
    return np.array([t, p, q])


def _dubins_LSR(alpha, beta, d):
    sa, sb, ca, cb = np.sin(alpha), np.sin(beta), np.cos(alpha), np.cos(beta)
    c_ab = np.cos(alpha - beta)
    p_sq = -2 + d * d + 2 * c_ab + 2 * d * (sa + sb)
    if p_sq < 0:
        return None
    p = np.sqrt(p_sq)
    tmp2 = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p)
    t = _mod2pi(-alpha + tmp2)
    q = _mod2pi(-_mod2pi(beta) + tmp2)
    return np.array([t, p, q])


def _dubins_RSL(alpha, beta, d):
    sa, sb, ca, cb = np.sin(alpha), np.sin(beta), np.cos(alpha), np.cos(beta)
    c_ab = np.cos(alpha - beta)
    p_sq = d * d - 2 + 2 * c_ab - 2 * d * (sa + sb)
    if p_sq < 0:
        return None
    p = np.sqrt(p_sq)
    tmp2 = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)
    t = _mod2pi(alpha - tmp2)
    q = _mod2pi(beta - tmp2)
    return np.array([t, p, q])


def _dubins_RLR(alpha, beta, d):
    sa, sb, ca, cb = np.sin(alpha), np.sin(beta), np.cos(alpha), np.cos(beta)
    c_ab = np.cos(alpha - beta)
    tmp = (6.0 - d * d + 2 * c_ab + 2 * d * (sa - sb)) / 8.0
    if abs(tmp) > 1:
        return None
    p = _mod2pi(2 * np.pi - np.arccos(tmp))
    t = _mod2pi(alpha - np.arctan2(ca - cb, d - sa + sb) + _mod2pi(p / 2.0))
    q = _mod2pi(alpha - beta - t + _mod2pi(p))
    return np.array([t, p, q])


def _dubins_LRL(alpha, beta, d):
    sa, sb, ca, cb = np.sin(alpha), np.sin(beta), np.cos(alpha), np.cos(beta)
    c_ab = np.cos(alpha - beta)
    tmp = (6.0 - d * d + 2 * c_ab + 2 * d * (-sa + sb)) / 8.0
    if abs(tmp) > 1:
        return None
    p = _mod2pi(2 * np.pi - np.arccos(tmp))
    t = _mod2pi(-alpha - np.arctan2(ca - cb, d + sa - sb) + p / 2.0)
    q = _mod2pi(_mod2pi(beta) - alpha - t + _mod2pi(p))
    return np.array([t, p, q])


_SOLVERS = [_dubins_LSL, _dubins_LSR, _dubins_RSL, _dubins_RSR, _dubins_RLR, _dubins_LRL]


def dubins_init(q0, q1, rho) -> Optional[DubinsPath]:
    """Compute the shortest Dubins path between two configurations.

    Args:
        q0: start (x, y, theta)
        q1: goal (x, y, theta)
        rho: minimum turning radius

    Returns:
        DubinsPath or None if no path found
    """
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)

    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    D = np.sqrt(dx * dx + dy * dy)
    d = D / rho

    if rho <= 0:
        return None

    theta = _mod2pi(np.arctan2(dy, dx))
    alpha = _mod2pi(q0[2] - theta)
    beta = _mod2pi(q1[2] - theta)

    best_cost = np.inf
    best_path = None

    for i, solver in enumerate(_SOLVERS):
        result = solver(alpha, beta, d)
        if result is not None:
            cost = result[0] + result[1] + result[2]
            if cost < best_cost:
                best_cost = cost
                best_path = DubinsPath(
                    qi=q0.copy(),
                    param=result.copy(),
                    rho=rho,
                    type=i,
                )

    return best_path


def _dubins_segment(t, qi, seg_type):
    """Compute configuration after traversing a single segment."""
    qt = np.zeros(3)
    if seg_type == L_SEG:
        qt[0] = qi[0] + np.sin(qi[2] + t) - np.sin(qi[2])
        qt[1] = qi[1] - np.cos(qi[2] + t) + np.cos(qi[2])
        qt[2] = qi[2] + t
    elif seg_type == R_SEG:
        qt[0] = qi[0] - np.sin(qi[2] - t) + np.sin(qi[2])
        qt[1] = qi[1] + np.cos(qi[2] - t) - np.cos(qi[2])
        qt[2] = qi[2] - t
    elif seg_type == S_SEG:
        qt[0] = qi[0] + np.cos(qi[2]) * t
        qt[1] = qi[1] + np.sin(qi[2]) * t
        qt[2] = qi[2]
    return qt


def dubins_path_length(path: DubinsPath) -> float:
    return (path.param[0] + path.param[1] + path.param[2]) * path.rho


def dubins_path_sample(path: DubinsPath, t: float) -> Optional[np.ndarray]:
    """Sample the path at distance t from the start.

    Returns:
        q = (x, y, theta) or None if t is out of range
    """
    length = dubins_path_length(path)
    if t < 0 or t >= length + EPSILON:
        return None

    t = min(t, length - EPSILON)
    tprime = t / path.rho

    qi = np.array([0.0, 0.0, path.qi[2]])
    types = DIRDATA[path.type]

    p1, p2 = path.param[0], path.param[1]
    q1 = _dubins_segment(p1, qi, types[0])
    q2 = _dubins_segment(p2, q1, types[1])

    if tprime < p1:
        q = _dubins_segment(tprime, qi, types[0])
    elif tprime < p1 + p2:
        q = _dubins_segment(tprime - p1, q1, types[1])
    else:
        q = _dubins_segment(tprime - p1 - p2, q2, types[2])

    q[0] = q[0] * path.rho + path.qi[0]
    q[1] = q[1] * path.rho + path.qi[1]
    q[2] = _mod2pi(q[2])
    return q


def dubins_path_sample_many(path: DubinsPath, step_size: float) -> np.ndarray:
    """Sample the path at regular intervals.

    Returns:
        z: (3, N) array of (x, y, theta) samples
    """
    length = dubins_path_length(path)
    samples = []
    t = 0.0
    while t < length:
        q = dubins_path_sample(path, t)
        if q is not None:
            samples.append(q)
        t += step_size
    return np.array(samples).T  # (3, N)
