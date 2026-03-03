from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planar_pushing_tools.config import OptsModel, set_contact_model_b
from planar_pushing_tools.push_learner import PushLearner

import demo_pusht_sysid_dubins as demo
from mpc.config import ClosedLoopRunConfig


def init_opts_model(T: float, rho: float, pt_local: np.ndarray) -> tuple[OptsModel, np.ndarray]:
    opts_model = OptsModel()
    opts_model.T = float(T)
    opts_model.rho = float(rho)
    opts_model.pt = pt_local.copy()
    opts_model.c = np.array(
        [-pt_local[1] / opts_model.rho, pt_local[0] / opts_model.rho, -1.0], dtype=float
    )
    b0 = opts_model.c / np.linalg.norm(opts_model.c)
    set_contact_model_b(opts_model, b0)
    return opts_model, b0


@dataclass
class OnlineLearnerState:
    learner: PushLearner
    accepted_updates: int = 0
    last_refresh_step: int = 0
    last_pose: np.ndarray | None = None


def run_initial_identification(
    env,
    info: dict,
    cfg: ClosedLoopRunConfig,
    pt_local: np.ndarray,
    opts_model: OptsModel,
) -> tuple[np.ndarray, int, int, dict]:
    learner = PushLearner(cfg.probe.learner_window, cfg.probe.probe_discount, opts_model)
    args = type(
        "ProbeArgs",
        (),
        {
            "n_probe": cfg.probe.n_probe,
            "probe_angle_deg": cfg.probe.probe_angle_deg,
            "probe_mag": cfg.probe.probe_mag,
        },
    )()
    b_est, flag_last, accepted, info = demo.run_sysid_probing(
        env,
        info,
        learner,
        opts_model,
        args,
        pt_action_local=pt_local,
        recorder=None,
    )
    return b_est, int(flag_last), int(accepted), info


def create_online_learner(cfg: ClosedLoopRunConfig, opts_model: OptsModel) -> OnlineLearnerState:
    learner = PushLearner(cfg.probe.learner_window, cfg.probe.probe_discount, opts_model)
    return OnlineLearnerState(learner=learner)


def maybe_refresh_model(
    learner_state: OnlineLearnerState,
    opts_model: OptsModel,
    step_idx: int,
    x_prev: np.ndarray,
    x_curr: np.ndarray,
    cfg: ClosedLoopRunConfig,
) -> tuple[bool, np.ndarray]:
    if not cfg.online_update.enabled:
        return False, opts_model.b
    if cfg.online_update.refresh_every_steps <= 0:
        return False, opts_model.b
    if step_idx - learner_state.last_refresh_step < cfg.online_update.refresh_every_steps:
        return False, opts_model.b
    if np.linalg.norm(x_curr[:2] - x_prev[:2]) < cfg.online_update.min_delta_xy:
        return False, opts_model.b

    flag_data = learner_state.learner.receive_data(x_curr, x_prev, opts_model)
    b_new, flag_train = learner_state.learner.train_svd(opts_model, flag_data)
    if flag_train <= 0:
        learner_state.last_refresh_step = step_idx
        return False, opts_model.b

    set_contact_model_b(opts_model, b_new)
    learner_state.accepted_updates += 1
    learner_state.last_refresh_step = step_idx
    return True, b_new

