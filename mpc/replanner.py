from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpc.model_update import init_opts_model
from planar_pushing_tools.config import set_contact_model_b
from planar_pushing_tools.push_planner_dubin import PushPlannerDubin

import demo_pusht_sysid_dubins as demo
from mpc.config import ClosedLoopRunConfig


@dataclass
class PlanBundle:
    x_plan: np.ndarray
    u_local: np.ndarray
    p_targets: np.ndarray
    pt_local: np.ndarray
    contact_label: str
    start_step: int
    used_until: int
    replan_reason: str


@dataclass
class ContactPlanCandidate:
    label: str
    plan: PlanBundle
    opts_model: object
    plan_length: int
    dist_final: float
    projected_progress: float


def build_plan(
    x_current: np.ndarray,
    goal_pose: np.ndarray,
    pt_local: np.ndarray,
    opts_model,
    cfg: ClosedLoopRunConfig,
    current_step: int,
    reason: str,
) -> PlanBundle:
    a_ls, b_ls = demo.estimate_dubins_limit_surface_from_b(opts_model.b, opts_model.c, opts_model.rho)
    planner = PushPlannerDubin(a_ls, b_ls, np.linalg.norm(pt_local), cfg.mu)

    goal_pose_plan = goal_pose.copy()
    goal_pose_plan[2] = demo.unwrap_angle_near_reference(float(goal_pose[2]), float(x_current[2]))
    step_size = max(float(planner.radius_turn / 50.0), float(cfg.replan.min_dubins_step_size))
    x_nominal, u_local = planner.plan(x_current, goal_pose_plan, step_size=step_size)
    p_targets = demo.pusher_path_from_block_traj(x_nominal, pt_local)

    if x_nominal.shape[1] > cfg.replan.horizon_steps + 1:
        keep = cfg.replan.horizon_steps + 1
        x_nominal = x_nominal[:, :keep]
        p_targets = p_targets[:, :keep]
        u_local = u_local[:, : keep - 1]

    used_until = min(cfg.replan.execute_chunk_steps, max(1, p_targets.shape[1] - 1))
    return PlanBundle(
        x_plan=x_nominal,
        u_local=u_local,
        p_targets=p_targets,
        pt_local=pt_local.copy(),
        contact_label="fixed",
        start_step=current_step,
        used_until=used_until,
        replan_reason=reason,
    )


def choose_adaptive_plan_across_contacts(
    x_current: np.ndarray,
    goal_pose: np.ndarray,
    contact_candidates: list[tuple[str, np.ndarray]],
    current_b: np.ndarray,
    cfg: ClosedLoopRunConfig,
    current_step: int,
    reason: str,
    T: float,
    recent_empirical_gain: float,
    active_contact_label: str,
    prefer_new_contact: bool,
    allow_contact_switch: bool,
) -> tuple[PlanBundle, object, dict]:
    candidates: list[ContactPlanCandidate] = []
    dist_current = float(np.linalg.norm(x_current[:2] - goal_pose[:2]))
    min_progress = float(cfg.replan.min_projected_progress_px)

    for label, pt_local in contact_candidates:
        opts_model, _ = init_opts_model(T, cfg.rho, pt_local)
        set_contact_model_b(opts_model, current_b)
        plan = build_plan(
            x_current=x_current,
            goal_pose=goal_pose,
            pt_local=pt_local,
            opts_model=opts_model,
            cfg=cfg,
            current_step=current_step,
            reason=reason,
        )
        plan.contact_label = label
        dist_final = float(np.linalg.norm(plan.x_plan[:2, -1] - goal_pose[:2]))
        projected_progress = dist_current - dist_final
        candidates.append(
            ContactPlanCandidate(
                label=label,
                plan=plan,
                opts_model=opts_model,
                plan_length=int(plan.x_plan.shape[1]),
                dist_final=dist_final,
                projected_progress=projected_progress,
            )
        )

    assert len(candidates) > 0
    shortest_length = min(item.plan_length for item in candidates)
    shortlist_margin = int(cfg.replan.shortlist_length_margin_steps)
    shortlist = [item for item in candidates if item.plan_length <= shortest_length + shortlist_margin]
    if len(shortlist) == 0:
        shortlist = list(candidates)

    progress_ok = [item for item in shortlist if item.projected_progress >= min_progress]
    if len(progress_ok) == 0:
        progress_ok = list(shortlist)

    pool = progress_ok
    if not allow_contact_switch:
        same = [item for item in pool if item.label == active_contact_label]
        if len(same) > 0:
            pool = same
    elif prefer_new_contact and cfg.replan.force_contact_switch_on_stagnation:
        switched = [item for item in pool if item.label != active_contact_label]
        if len(switched) > 0:
            pool = switched

    recovery_mode = recent_empirical_gain < float(cfg.replan.min_empirical_gain_px)
    strategy = "shortest_then_progress"
    if recovery_mode or prefer_new_contact:
        strategy = "progress_first"
        selected = max(pool, key=lambda item: (item.projected_progress, -item.plan_length))
    else:
        selected = min(pool, key=lambda item: (item.plan_length, -item.projected_progress, item.dist_final))

    diagnostics = {
        "strategy": strategy,
        "recent_empirical_gain": float(recent_empirical_gain),
        "candidate_count": int(len(candidates)),
        "shortlist_count": int(len(shortlist)),
        "progress_ok_count": int(len([item for item in shortlist if item.projected_progress >= min_progress])),
        "selected_label": selected.label,
        "selected_plan_length": int(selected.plan_length),
        "selected_projected_progress": float(selected.projected_progress),
        "selected_dist_final": float(selected.dist_final),
        "min_projected_progress_threshold": float(min_progress),
        "prefer_new_contact": bool(prefer_new_contact),
        "allow_contact_switch": bool(allow_contact_switch),
    }
    return selected.plan, selected.opts_model, diagnostics


def should_replan(
    plan: PlanBundle | None,
    within_plan_step: int,
    contact_loss_streak: int,
    action_clipped: bool,
    tracking_err_px: float,
    cfg: ClosedLoopRunConfig,
) -> tuple[bool, str]:
    if plan is None:
        return True, "init"
    if within_plan_step >= plan.used_until:
        return True, "chunk_exhausted"
    if contact_loss_streak >= cfg.replan.replan_on_contact_loss_streak:
        return True, "contact_loss"
    if cfg.replan.replan_on_clipped_action and action_clipped:
        return True, "clipped_action"
    if tracking_err_px > cfg.replan.replan_on_tracking_err_px:
        return True, "tracking_error"
    return False, "follow_plan"

