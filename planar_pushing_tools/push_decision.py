"""PushDecision: state machine for probing/planning. Port of wrapper/PushDecision.m."""

import numpy as np
from dataclasses import dataclass
from .config import set_contact_model_b


@dataclass
class DecisionInfo:
    besti: np.ndarray = None
    learned: bool = False
    exitflag: int = 1
    bnorm: float = 0.0


class PushDecision:
    def __init__(self, planner, learner, x0, data_NH, horizon):
        """
        Args:
            planner: PushPlanner instance
            learner: PushLearner instance
            x0: initial state (3,)
            data_NH: learning horizon
            horizon: planning horizon
        """
        self.planner = planner
        self.learner = learner
        self.state = 0  # 0: probing and learning, 1: planning motion

        self.x = x0.copy()
        self.urand = np.array([planner.u_limit_ang[0], planner.u_limit_mag[1]])

        self.check_threshold_dx = 0.002
        self.check_threshold_dist = 0.1
        self.check_threshold_xdev = 0.01

        self.counter = 0
        self.nt = 0
        self.NT = data_NH
        self.RH = False
        self.horizon = horizon

        self.planner.ddp_para.detail = 1

    def decide(self, xn, opts_model):
        """Main decision function.

        Args:
            xn: current state (3,)
            opts_model: OptsModel instance

        Returns:
            u: control (2,)
            info: DecisionInfo
        """
        besti, flag_learned = self._learn(xn, opts_model)

        info = DecisionInfo()
        info.besti = besti.copy() if isinstance(besti, np.ndarray) else opts_model.b.copy()

        if self.state == 0:
            # use random action to probe and update model parameter
            self.nt = 0
            if self.counter <= 0:
                if flag_learned >= 0:
                    self.counter += 1
                self.urand = np.array([-self.urand[0], self.urand[1]])
                u = self.urand.copy()
            elif self.counter == 1:
                set_contact_model_b(opts_model, besti)
                info.learned = True
                u, info.bnorm, info.exitflag = self._plan(xn, True, opts_model)
                self.counter = 0
                self.state = 1
        elif self.state == 1:
            self.nt += 1
            flag = self._check_prediction(xn, opts_model)
            if flag:
                u, info.bnorm, info.exitflag = self._plan(xn, self.RH, opts_model)
            else:
                self.urand = np.array([-self.urand[0], self.urand[1]])
                u = self.urand.copy()
                self.state = 0

        self.x = xn.copy()
        return u, info

    def _learn(self, xn, opts_model):
        """Learn contact model from recent data."""
        besti = opts_model.b.copy()
        flag = self.learner.receive_data(xn, self.x, opts_model)
        bnew, flag = self.learner.train_svd(opts_model, flag)
        if flag >= 0:
            besti = bnew
        return besti, flag

    def _plan(self, xn, replan, opts_model):
        """Plan trajectory and return control.

        Returns:
            u: control (2,)
            bnorm: norm of feedback gain
            exitflag: solver exit flag
        """
        if replan:
            self.planner.set_initial_pose(xn, opts_model)
            unom = self.planner.unom
            urefrand = np.random.rand(2, self.NT)
            urefrand[0, :] = (np.random.rand(self.NT) - 0.5) * 2 * self.planner.u_limit_ang[1]
            urefrand[1, :] = np.random.rand(self.NT) * self.planner.u_limit_mag[1]
            uref = np.hstack([unom[:, self.NT:], urefrand])
            self.NT = 1
            info = self.planner.train_controller(uref, opts_model)
            exitflag = info.exitflag
            print(f"DDP iter: {info.iter}")
            if exitflag < 0:
                return np.zeros(2), 0.0, exitflag
        else:
            exitflag = 1
            self.NT += 1
            if self.NT >= self.horizon - 1:
                return np.zeros(2), 0.0, -1

        u, beta_i = self.planner.control_output_i(xn, self.NT)
        bnorm = np.linalg.norm(beta_i)
        return u, bnorm, exitflag

    def _check_prediction(self, xn, opts_model):
        """Check if model prediction matches reality."""
        dx = xn - self.x

        # Check model prediction
        theta = self.x[2]
        Rot = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,            0.0,           1.0],
        ])
        Y = np.diag([1.0, 1.0, opts_model.rho]) @ np.linalg.inv(Rot) @ dx

        norm_Y = np.linalg.norm(Y)
        if norm_Y > 1e-10:
            dist = abs(opts_model.b @ Y) / norm_Y
            if dist > self.check_threshold_dist:
                return False

        # Check deviation from nominal trajectory
        if self.planner.xnom is not None and self.nt < self.planner.xnom.shape[1]:
            xnom_i = self.planner.xnom[:, self.nt]
            dev = xnom_i - xn
            dev[2] *= opts_model.rho
            if np.linalg.norm(dev) > self.check_threshold_xdev:
                return False

        return True
