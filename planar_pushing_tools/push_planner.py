"""PushPlanner: trajectory planning via DDP. Port of wrapper/PushPlanner.m."""

import numpy as np
from .ddp_solver import DDPPara, ddp_solve


class PushPlanner:
    def __init__(self, N, u_limit_ang, u_limit_mag):
        """
        Args:
            N: horizon length
            u_limit_ang: (2,) [lower, upper] angular limits in radians
            u_limit_mag: (2,) [lower, upper] magnitude limits
        """
        self.N = N
        self.u_limit_ang = np.array(u_limit_ang, dtype=float)
        self.u_limit_mag = np.array(u_limit_mag, dtype=float)

        # DDP solver parameters (overrides from PushPlanner.m)
        self.ddp_para = DDPPara()
        self.ddp_para.detail = 1
        self.ddp_para.Ci_reg_init = 1.0
        self.ddp_para.Ci_reg_Max = 1e12
        self.ddp_para.Ci_reg_fac = 3.0
        self.ddp_para.Ci_reg_localloop = 3
        self.ddp_para.maxIter = 200
        self.ddp_para.tolGrad = 1e-4
        self.ddp_para.MinCostChange = 0.1

        # Problem description
        self.xinit = None
        self.xgoal = None

        # Solution
        self.xnom = None
        self.unom = None
        self.beta = None

    def set_initial_pose(self, xinit, opts_model=None):
        self.xinit = xinit.copy()
        if opts_model is not None:
            opts_model.xinit = xinit.copy()

    def set_goal_pose(self, xgoal):
        self.xgoal = xgoal.copy()

    def train_controller(self, uref, opts_model):
        """Run DDP twice: exploration then tracking.

        Args:
            uref: initial control sequence (2, N) or None for random
            opts_model: OptsModel instance

        Returns:
            info: DDPInfo from the second (tracking) pass
        """
        if uref is None:
            self.unom = np.random.rand(2, self.N)
        else:
            self.unom = uref.copy()

        # Control bounds
        ulb = np.vstack([
            self.u_limit_ang[0] * np.ones(self.N),
            self.u_limit_mag[0] * np.ones(self.N),
        ])
        uub = np.vstack([
            self.u_limit_ang[1] * np.ones(self.N),
            self.u_limit_mag[1] * np.ones(self.N),
        ])

        # First pass: exploration with tighter bounds
        c = 0.8
        ulb_c = c * ulb
        uub_c = uub.copy()

        self.xnom, self.unom, _, _, info0 = ddp_solve(
            self.xinit, self.unom, opts_model, ulb_c, uub_c, self.ddp_para
        )

        # Second pass: tracking the planned trajectory
        opts_model.xref = self.xnom.copy()
        self.xnom, self.unom, _, self.beta, info = ddp_solve(
            self.xinit, self.unom, opts_model, ulb, uub, self.ddp_para
        )

        return info

    def control_output_i(self, x, i):
        """Feedback control at timestep i.

        Args:
            x: current state (3,)
            i: timestep index (0-based)

        Returns:
            u: control (2,)
            beta_i: feedback gain matrix (2, 3)
        """
        deltaX = x - self.xnom[:, i]
        u = self.unom[:, i] + self.beta[:, :, i] @ deltaX
        u[0] = np.clip(u[0], self.u_limit_ang[0], self.u_limit_ang[1])
        u[1] = np.clip(u[1], self.u_limit_mag[0], self.u_limit_mag[1])
        beta_i = self.beta[:, :, i]
        return u, beta_i
