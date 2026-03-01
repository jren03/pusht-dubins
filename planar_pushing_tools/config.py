"""Model configuration replacing MATLAB global opts_model."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OptsModel:
    # State cost weights
    Q: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Qf: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Huber loss parameters
    ep: np.ndarray = field(default_factory=lambda: np.zeros(3))
    epf: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Control cost
    R: np.ndarray = field(default_factory=lambda: np.zeros(2))

    # Time step
    T: float = 0.1

    # State / control dimensions
    NX: int = 3
    NU: int = 2

    # Contact geometry
    rho: float = 0.02
    pt: np.ndarray = field(default_factory=lambda: np.zeros(2))
    c: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Contact model
    b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    D: Optional[np.ndarray] = None
    D_inv: Optional[np.ndarray] = None

    # Initial state and reference trajectory
    xinit: np.ndarray = field(default_factory=lambda: np.zeros(3))
    xref: Optional[np.ndarray] = None  # (3, NH)


def get_contact_model_b(opts: OptsModel, b: np.ndarray):
    """Compute D and D_inv from contact model parameter b.

    Port of GetContactModelb.m:
        B     = [I2, [-pt(2)/rho; pt(1)/rho]]
        D     = [B; b']
        D_inv = inv(D)
        D_inv(3,:) = D_inv(3,:) / rho
    """
    B = np.array([
        [1.0, 0.0, -opts.pt[1] / opts.rho],
        [0.0, 1.0,  opts.pt[0] / opts.rho],
    ])
    D = np.vstack([B, b.reshape(1, 3)])
    D_inv = np.linalg.inv(D)
    D_inv[2, :] = D_inv[2, :] / opts.rho
    return D, D_inv


def set_contact_model_b(opts: OptsModel, b: np.ndarray):
    """Set contact model parameter b and update D, D_inv.

    Port of SetContactModelb.m.
    """
    D, D_inv = get_contact_model_b(opts, b)
    opts.b = b.copy()
    opts.D_inv = D_inv
    opts.D = D
