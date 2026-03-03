from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np

from mpc.config import ClosedLoopRunConfig
from mpc.logging import write_csv, write_json, write_jsonl
from mpc.metrics import metric_row, summarize_rows
from mpc.model_update import create_online_learner, init_opts_model, maybe_refresh_model, run_initial_identification
from mpc.replanner import PlanBundle, build_plan, choose_adaptive_plan_across_contacts, should_replan
from planar_pushing_tools.demo_pusht_sysid_dubins_viz import MP4Recorder, save_frames_to_mp4, save_start_comparison_image

import demo_pusht_sysid_dubins as demo


@dataclass
class EpisodeResult:
    rows: list[dict]
    summary: dict
    plan_reasons: list[str]
    pt_local: np.ndarray


def run_closed_loop_episode(
    cfg: ClosedLoopRunConfig,
    pt_local: np.ndarray,
    run_tag: str,
    save_artifacts: bool,
    contact_candidates: list[tuple[str, np.ndarray]] | None = None,
) -> EpisodeResult:
    demo.ensure_gym_pusht_importable()
    __import__("gym_pusht")
    import gymnasium as gym

    np.random.seed(cfg.seed)
    need_frames = bool(save_artifacts and (cfg.save_mp4_path or cfg.save_plot_path))
    render_mode = "rgb_array" if need_frames else cfg.render_mode
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=render_mode)
    recorder = MP4Recorder(enabled=need_frames)
    try:
        reset_options = None
        if cfg.reset_to_state is not None:
            reset_options = {"reset_to_state": np.asarray(cfg.reset_to_state, dtype=float)}
        _, info = env.reset(seed=cfg.seed, options=reset_options)
        recorder.capture(env)

        T = 1.0 / env.unwrapped.control_hz
        coverage_threshold = float(env.unwrapped.success_threshold)
        goal_pose = np.asarray(info["goal_pose"], dtype=float).copy()

        opts_model, b0 = init_opts_model(T, cfg.rho, pt_local)
        active_contact_label = "fixed"
        info = demo.place_pusher_at_local_contact_without_step(env, info, pt_local, recorder=recorder)
        b_est, flag_last, accepted, info = run_initial_identification(env, info, cfg, pt_local, opts_model)
        if flag_last < 0:
            b_est = b0
            from planar_pushing_tools.config import set_contact_model_b

            set_contact_model_b(opts_model, b_est)

        learner_state = create_online_learner(cfg, opts_model)
        learner_state.last_pose = demo.block_pose_from_info(info)

        rows: list[dict] = []
        plan_reasons: list[str] = []
        plan: PlanBundle | None = None
        within_plan_step = 0
        replan_count = 0
        contact_switch_count = 0
        contact_loss_streak = 0
        last_action_clipped = False
        last_tracking_err = 0.0
        replan_stagnation_count = 0
        selection_diagnostics_rows: list[dict] = []
        selection_strategy_counts: dict[str, int] = {}
        selected_projected_progress_values: list[float] = []
        recent_gain_values: list[float] = []
        rejected_progress_total = 0
        active_selection_diag: dict | None = None
        last_contact_switch_step = -1_000_000
        switch_cooldown_blocks = 0
        boundary_shape_count = 0
        boundary_shift_values: list[float] = []
        reacquire_remaining_steps = 0
        reacquire_success_count = 0
        reacquire_fail_count = 0
        reacquire_attempt_count = 0

        x_prev = demo.block_pose_from_info(info)
        p_prev = demo.agent_pos_from_info(info)
        x_hist = [x_prev.copy()]
        p_hist = [p_prev.copy()]

        for step_idx in range(cfg.max_rollout_steps):
            recent_gain = recent_goal_distance_gain(rows, cfg.replan.empirical_gain_window_steps)
            stagnating = is_coverage_stagnating(
                rows,
                cfg.replan.stagnation_window_steps,
                cfg.replan.stagnation_min_delta_coverage,
            )
            must_replan, reason = should_replan(
                plan,
                within_plan_step,
                contact_loss_streak,
                last_action_clipped,
                last_tracking_err,
                cfg,
            )
            if stagnating:
                must_replan = True
                reason = "coverage_stagnation"
                replan_stagnation_count += 1
            if must_replan:
                if contact_candidates is None:
                    plan = build_plan(
                        x_current=x_prev,
                        goal_pose=goal_pose,
                        pt_local=pt_local,
                        opts_model=opts_model,
                        cfg=cfg,
                        current_step=step_idx,
                        reason=reason,
                    )
                else:
                    allow_contact_switch = (
                        step_idx - last_contact_switch_step
                    ) >= int(cfg.replan.contact_switch_cooldown_steps)
                    if stagnating and not allow_contact_switch:
                        switch_cooldown_blocks += 1
                    plan, opts_model_new, selection_diag = choose_adaptive_plan_across_contacts(
                        x_current=x_prev,
                        goal_pose=goal_pose,
                        contact_candidates=contact_candidates,
                        current_b=opts_model.b.copy(),
                        cfg=cfg,
                        current_step=step_idx,
                        reason=reason,
                        T=T,
                        recent_empirical_gain=recent_gain,
                        active_contact_label=active_contact_label,
                        prefer_new_contact=stagnating,
                        allow_contact_switch=allow_contact_switch,
                    )
                    active_selection_diag = dict(selection_diag)
                    selection_diagnostics_rows.append(
                        {
                            "step": int(step_idx),
                            "reason": reason,
                            "active_contact_before": active_contact_label,
                            **selection_diag,
                        }
                    )
                    strategy = str(selection_diag["strategy"])
                    selection_strategy_counts[strategy] = selection_strategy_counts.get(strategy, 0) + 1
                    selected_projected_progress_values.append(float(selection_diag["selected_projected_progress"]))
                    recent_gain_values.append(float(selection_diag["recent_empirical_gain"]))
                    rejected_progress_total += int(selection_diag["shortlist_count"]) - int(
                        selection_diag["progress_ok_count"]
                    )
                    switched = not np.allclose(plan.pt_local, pt_local, atol=1e-9)
                    if switched:
                        contact_switch_count += 1
                        last_contact_switch_step = step_idx
                        reacquire_remaining_steps = int(max(0, cfg.replan.reacquire_steps_after_switch))
                        reacquire_attempt_count += 1
                        pt_local = plan.pt_local.copy()
                        active_contact_label = plan.contact_label
                        info = demo.place_pusher_at_local_contact_without_step(
                            env, info, pt_local, recorder=recorder
                        )
                        x_prev = demo.block_pose_from_info(info)
                        p_prev = demo.agent_pos_from_info(info)
                    opts_model = opts_model_new
                within_plan_step = 0
                replan_count += 1
                plan_reasons.append(reason)

            assert plan is not None
            target_idx = min(within_plan_step + 1, plan.p_targets.shape[1] - 1)
            action_plan_raw = plan.p_targets[:, target_idx]
            action_shaped = shape_action_toward_interior(
                action_plan_raw,
                float(cfg.replan.action_boundary_margin_px),
            )
            if reacquire_remaining_steps > 0:
                action_shaped = blend_reacquire_action(
                    action_shaped=action_shaped,
                    p_prev=p_prev,
                    alpha=float(cfg.replan.reacquire_alpha),
                )
            action_cmd = demo.clip_action(action_shaped)
            boundary_shift = float(np.linalg.norm(action_shaped - action_plan_raw))
            if boundary_shift > 1e-9:
                boundary_shape_count += 1
            boundary_shift_values.append(boundary_shift)

            _, _, terminated, truncated, info = env.step(action_cmd)
            recorder.capture(env)

            x_true = demo.block_pose_from_info(info)
            p_true = demo.agent_pos_from_info(info)
            coverage = float(info["coverage"])
            n_contacts = int(info["n_contacts"])
            in_contact = n_contacts > 0
            contact_loss_streak = 0 if in_contact else contact_loss_streak + 1
            if reacquire_remaining_steps > 0:
                if n_contacts >= int(cfg.replan.reacquire_contact_target_n):
                    reacquire_success_count += 1
                    reacquire_remaining_steps = 0
                else:
                    reacquire_remaining_steps -= 1
                    if reacquire_remaining_steps == 0:
                        reacquire_fail_count += 1

            x_plan_now = plan.x_plan[:, target_idx]
            p_target_now = plan.p_targets[:, target_idx]
            model_pos_err = float(np.linalg.norm(x_true[:2] - x_plan_now[:2]))
            row = metric_row(
                step_idx=step_idx,
                phase="closed_loop",
                x_true=x_true,
                p_true=p_true,
                x_plan=x_plan_now,
                p_target=p_target_now,
                goal_pose=goal_pose,
                action_cmd=action_cmd,
                action_raw=action_plan_raw,
                n_contacts=n_contacts,
                coverage=coverage,
                coverage_threshold=coverage_threshold,
                replan_reason=plan.replan_reason if within_plan_step == 0 else "follow_plan",
                replan_count=replan_count,
                contact_loss_streak=contact_loss_streak,
                model_pos_err=model_pos_err,
            )
            row["active_contact_label"] = active_contact_label
            row["active_contact_pt_x"] = float(pt_local[0])
            row["active_contact_pt_y"] = float(pt_local[1])
            row["contact_switch_count"] = int(contact_switch_count)
            row["boundary_action_shaped"] = bool(boundary_shift > 1e-9)
            row["boundary_action_shift_px"] = float(boundary_shift)
            row["action_x_shaped"] = float(action_shaped[0])
            row["action_y_shaped"] = float(action_shaped[1])
            row["reacquire_active"] = bool(reacquire_remaining_steps > 0)
            row["reacquire_remaining_steps"] = int(reacquire_remaining_steps)
            row["recent_empirical_gain"] = float(recent_gain)
            row["stagnating_now"] = bool(stagnating)
            if active_selection_diag is None:
                row["selection_strategy"] = "none"
                row["selection_projected_progress"] = 0.0
                row["selection_candidate_count"] = 0
                row["selection_shortlist_count"] = 0
                row["selection_progress_ok_count"] = 0
            else:
                row["selection_strategy"] = str(active_selection_diag["strategy"])
                row["selection_projected_progress"] = float(active_selection_diag["selected_projected_progress"])
                row["selection_candidate_count"] = int(active_selection_diag["candidate_count"])
                row["selection_shortlist_count"] = int(active_selection_diag["shortlist_count"])
                row["selection_progress_ok_count"] = int(active_selection_diag["progress_ok_count"])
            rows.append(row)
            x_hist.append(x_true.copy())
            p_hist.append(p_true.copy())

            _updated, _ = maybe_refresh_model(
                learner_state=learner_state,
                opts_model=opts_model,
                step_idx=step_idx,
                x_prev=x_prev,
                x_curr=x_true,
                cfg=cfg,
            )

            last_action_clipped = bool(row["action_clipped"])
            last_tracking_err = float(row["tracking_pos_err"])
            x_prev = x_true
            p_prev = p_true
            within_plan_step += 1

            if terminated or truncated:
                break
            if coverage >= coverage_threshold:
                break

        summary = summarize_rows(rows)
        summary["seed"] = int(cfg.seed)
        summary["pt_local"] = [float(pt_local[0]), float(pt_local[1])]
        summary["rho"] = float(cfg.rho)
        summary["mu"] = float(cfg.mu)
        summary["active_contact_label"] = active_contact_label
        summary["accepted_probe_updates"] = int(accepted)
        summary["accepted_online_updates"] = int(learner_state.accepted_updates)
        summary["contact_switch_count"] = int(contact_switch_count)
        summary["replan_stagnation_count"] = int(replan_stagnation_count)
        summary["switch_cooldown_blocks"] = int(switch_cooldown_blocks)
        summary["selection_strategy_counts"] = selection_strategy_counts
        summary["mean_selected_projected_progress"] = float(
            np.mean(np.asarray(selected_projected_progress_values, dtype=float))
            if len(selected_projected_progress_values) > 0
            else 0.0
        )
        summary["mean_recent_empirical_gain"] = float(
            np.mean(np.asarray(recent_gain_values, dtype=float)) if len(recent_gain_values) > 0 else 0.0
        )
        summary["rejected_progress_candidates_total"] = int(rejected_progress_total)
        summary["boundary_shaping_fraction"] = float(boundary_shape_count / max(1, len(rows)))
        summary["mean_boundary_shift_px"] = float(np.mean(np.asarray(boundary_shift_values, dtype=float)))
        summary["reacquire_attempt_count"] = int(reacquire_attempt_count)
        summary["reacquire_success_count"] = int(reacquire_success_count)
        summary["reacquire_fail_count"] = int(reacquire_fail_count)
        summary["reacquire_success_fraction"] = float(
            float(reacquire_success_count) / max(1, float(reacquire_attempt_count))
        )
        summary["likely_culprit"] = infer_likely_culprit(summary)

        if save_artifacts:
            write_jsonl(cfg.save_debug_log_path, rows)
            write_csv(cfg.save_debug_log_path.replace(".jsonl", ".csv"), rows)
            write_json(cfg.save_summary_path, summary)
            write_json(
                cfg.save_debug_log_path.replace(".jsonl", "_replan_decisions.json"),
                {"rows": selection_diagnostics_rows},
            )
            if cfg.save_mp4_path:
                mp4_fps = (
                    float(cfg.mp4_fps)
                    if cfg.mp4_fps is not None
                    else float(env.unwrapped.metadata["render_fps"])
                )
                save_frames_to_mp4(recorder.frames, cfg.save_mp4_path, mp4_fps)
            if cfg.save_plot_path:
                x_arr = np.stack(x_hist, axis=1)
                p_arr = np.stack(p_hist, axis=1)
                save_start_comparison_image(
                    pusht_start_frame_rgb=recorder.frames[0],
                    x_traj=x_arr,
                    p_pusher_traj=p_arr,
                    goal_pose=goal_pose,
                    get_tee_polygons=demo.tee_polygons_world,
                    save_path=cfg.save_plot_path,
                )
        return EpisodeResult(rows=rows, summary=summary, plan_reasons=plan_reasons, pt_local=pt_local.copy())
    finally:
        env.close()


def evaluate_contact_point(cfg: ClosedLoopRunConfig, pt_local: np.ndarray, label: str) -> dict:
    quick_cfg = copy.deepcopy(cfg)
    quick_cfg.max_rollout_steps = min(quick_cfg.max_rollout_steps, 40)
    result = run_closed_loop_episode(
        quick_cfg,
        pt_local,
        run_tag=f"eval_{label}",
        save_artifacts=False,
        contact_candidates=[(label, pt_local.copy())],
    )
    summary = dict(result.summary)
    summary["label"] = label
    summary["pt_local"] = [float(pt_local[0]), float(pt_local[1])]
    return summary


def recent_goal_distance_gain(rows: list[dict], window_steps: int) -> float:
    if len(rows) == 0:
        return 0.0
    count = max(1, int(window_steps))
    tail = rows[-count:]
    first = float(tail[0]["goal_center_dist"])
    last = float(tail[-1]["goal_center_dist"])
    return first - last


def is_coverage_stagnating(rows: list[dict], window_steps: int, min_delta: float) -> bool:
    count = max(1, int(window_steps))
    if len(rows) < count:
        return False
    tail = rows[-count:]
    start = float(tail[0]["coverage"])
    end = float(tail[-1]["coverage"])
    return (end - start) < float(min_delta)


def infer_likely_culprit(summary: dict) -> str:
    if bool(summary["coverage_pass_max"]):
        return "none"
    if int(summary["reacquire_attempt_count"]) > 0 and float(summary["reacquire_success_fraction"]) < 0.3:
        return "post_switch_contact_reacquire_failure"
    if float(summary["switch_cooldown_blocks"]) > 0 and int(summary["contact_switch_count"]) <= 1:
        return "contact_switch_cooldown_overconstrained"
    if float(summary["boundary_shaping_fraction"]) > 0.2 and float(summary["clipping_fraction"]) > 0.2:
        return "boundary_fighting_even_after_shaping"
    if float(summary["in_contact_fraction"]) > 0.9 and float(summary["clipping_fraction"]) < 0.05:
        if float(summary["mean_selected_projected_progress"]) < 1.0:
            return "planner_model_mismatch_low_projected_progress"
        if int(summary["replan_stagnation_count"]) > 0 and int(summary["contact_switch_count"]) == 0:
            return "contact_strategy_stuck_without_switch"
        return "stable_contact_but_low_task_progress"
    if float(summary["clipping_fraction"]) >= 0.1:
        return "action_clipping_or_workspace_boundary"
    if float(summary["in_contact_fraction"]) < 0.5:
        return "contact_breakage_or_contact_targeting"
    return "mixed_factors_needs_replan_diagnostics"


def shape_action_toward_interior(action: np.ndarray, margin_px: float) -> np.ndarray:
    lo = float(max(0.0, margin_px))
    hi = float(512.0 - lo)
    shaped = action.copy()
    shaped[0] = float(np.clip(shaped[0], lo, hi))
    shaped[1] = float(np.clip(shaped[1], lo, hi))
    return shaped


def blend_reacquire_action(action_shaped: np.ndarray, p_prev: np.ndarray, alpha: float) -> np.ndarray:
    w = float(np.clip(alpha, 0.0, 1.0))
    return w * action_shaped + (1.0 - w) * p_prev

