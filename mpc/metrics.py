from __future__ import annotations

import numpy as np

import demo_pusht_sysid_dubins as demo


def metric_row(
    step_idx: int,
    phase: str,
    x_true: np.ndarray,
    p_true: np.ndarray,
    x_plan: np.ndarray,
    p_target: np.ndarray,
    goal_pose: np.ndarray,
    action_cmd: np.ndarray,
    action_raw: np.ndarray,
    n_contacts: int,
    coverage: float,
    coverage_threshold: float,
    replan_reason: str,
    replan_count: int,
    contact_loss_streak: int,
    model_pos_err: float,
) -> dict:
    return {
        "step": int(step_idx),
        "phase": phase,
        "x_true": float(x_true[0]),
        "y_true": float(x_true[1]),
        "theta_true": float(x_true[2]),
        "x_plan": float(x_plan[0]),
        "y_plan": float(x_plan[1]),
        "theta_plan": float(x_plan[2]),
        "p_true_x": float(p_true[0]),
        "p_true_y": float(p_true[1]),
        "p_target_x": float(p_target[0]),
        "p_target_y": float(p_target[1]),
        "action_x_raw": float(action_raw[0]),
        "action_y_raw": float(action_raw[1]),
        "action_x_cmd": float(action_cmd[0]),
        "action_y_cmd": float(action_cmd[1]),
        "action_clipped": bool(np.any(np.abs(action_cmd - action_raw) > 1e-9)),
        "coverage": float(coverage),
        "coverage_threshold": float(coverage_threshold),
        "success": bool(coverage >= coverage_threshold),
        "n_contacts": int(n_contacts),
        "in_contact": bool(n_contacts > 0),
        "contact_loss_streak": int(contact_loss_streak),
        "tracking_pos_err": float(np.linalg.norm(x_true[:2] - x_plan[:2])),
        "tracking_ang_err": float(np.abs(demo.wrap_to_pi(x_true[2] - x_plan[2]))),
        "tracking_pusher_err": float(np.linalg.norm(p_true - p_target)),
        "goal_center_dist": float(np.linalg.norm(x_true[:2] - goal_pose[:2])),
        "replan_reason": replan_reason,
        "replan_count": int(replan_count),
        "model_pos_err": float(model_pos_err),
    }


def summarize_rows(rows: list[dict]) -> dict:
    coverage_values = np.asarray([float(row["coverage"]) for row in rows], dtype=float)
    clipped = np.asarray([float(bool(row["action_clipped"])) for row in rows], dtype=float)
    in_contact = np.asarray([float(bool(row["in_contact"])) for row in rows], dtype=float)
    tracking = np.asarray([float(row["tracking_pos_err"]) for row in rows], dtype=float)
    model_err = np.asarray([float(row["model_pos_err"]) for row in rows], dtype=float)
    contact_streak = np.asarray([float(row["contact_loss_streak"]) for row in rows], dtype=float)
    replans = np.asarray([float(row["replan_count"]) for row in rows], dtype=float)
    threshold = float(rows[0]["coverage_threshold"])

    return {
        "steps": len(rows),
        "coverage_start": float(coverage_values[0]),
        "coverage_max": float(np.max(coverage_values)),
        "coverage_final": float(coverage_values[-1]),
        "coverage_threshold": threshold,
        "coverage_pass_final": bool(coverage_values[-1] >= threshold),
        "coverage_pass_max": bool(np.max(coverage_values) >= threshold),
        "coverage_progress": float(np.max(coverage_values) - coverage_values[0]),
        "clipping_fraction": float(np.mean(clipped)),
        "in_contact_fraction": float(np.mean(in_contact)),
        "max_contact_loss_streak": int(np.max(contact_streak)),
        "mean_tracking_pos_err": float(np.mean(tracking)),
        "final_tracking_pos_err": float(tracking[-1]),
        "mean_model_pos_err": float(np.mean(model_err)),
        "final_model_pos_err": float(model_err[-1]),
        "replan_count_final": int(replans[-1]),
    }

