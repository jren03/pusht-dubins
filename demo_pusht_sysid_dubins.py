"""Push-T integration demo: sys-id, Dubins rollout, and environment-metric progress.

This script intentionally excludes DDP and focuses on:
1) Seeded Push-T reset
2) Contact probing + system identification (weighted SVD)
3) Dubins planning from identified contact model
4) Expected model rollout visualization
5) Open-loop execution in Push-T with coverage-based progress evaluation
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import shapely.geometry as sg

from planar_pushing_tools.config import OptsModel, set_contact_model_b
from planar_pushing_tools.demo_pusht_sysid_dubins_viz import (
    MP4Recorder,
    save_frames_to_mp4,
    save_start_comparison_image,
)
from planar_pushing_tools.model import f_
from planar_pushing_tools.push_learner import PushLearner
from planar_pushing_tools.push_planner_dubin import PushPlannerDubin

# ------------------------- Constant Config -------------------------
# Execution
SEED = 2
RESET_TO_STATE = None  # e.g. [agent_x, agent_y, block_x, block_y, block_theta]
RENDER_MODE = "rgb_array"  # "rgb_array" or "human"
SHOW_PLOT = False

# Identification / model
N_PROBE = 40
PROBE_ANGLE_DEG = 45.0
PROBE_MAG = 12.0
PROBE_DISCOUNT = 0.9
RHO = 60.0  # Value in pixel space
# Same weighted-SVD logic as demo_planars.py, but with a shorter rolling window
# to allow multiple accepted updates during a longer probing phase in Push-T.
LEARNER_WINDOW = 12

# Best triage contact point (~0.56 max coverage in full-horizon sweeps):
# left-side contact near the upper crossbar region.
PT_LOCAL = np.array([-75.0, 25.0], dtype=float)

MU = 0.2

# Optional contact-point search before final rollout visualization.
# If enabled, evaluate side contact points (excluding corners) and pick the
# shortest planned frame-count candidate, then run/save that chosen case.
SEARCH_CONTACT_POINTS = True
CONTACT_POINTS_PER_SIDE = 7
CONTACT_CORNER_MARGIN = 5.0
CONTACT_SIDES = ("left", "right", "top", "bottom")

# Rollout horizon
MAX_ROLLOUT_STEPS = 0  # <=0 means full Dubins plan
# Avoid over-dense Dubins samples when radius_turn is small; in Push-T this can
# produce tiny target updates and negligible object motion in open loop.
MIN_DUBINS_STEP_SIZE = 1.2

# Outputs (saved to current working directory)
SAVE_PLOT = True
SAVE_ROLLOUT_MP4 = True
SAVE_DEBUG_LOG = True
OUTPUT_PLOT_PATH = "pusht_start_comparison.png"
OUTPUT_ROLLOUT_MP4_PATH = "pusht_rollout.mp4"
OUTPUT_DEBUG_LOG_PATH = "pusht_rollout_debug.txt"
OUTPUT_FPS = None  # None -> use env render_fps


def ensure_gym_pusht_importable():
    repo_root = Path(__file__).resolve().parent
    local_pkg = repo_root / "gym-pusht"
    if str(local_pkg) not in sys.path:
        sys.path.insert(0, str(local_pkg))


def wrap_to_pi(angle: np.ndarray | float):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_angle_near_reference(target_angle: float, reference_angle: float) -> float:
    return float(reference_angle + wrap_to_pi(target_angle - reference_angle))


def rot2(theta: float) -> np.ndarray:
    ct = float(np.cos(theta))
    st = float(np.sin(theta))
    return np.array([[ct, -st], [st, ct]], dtype=float)


def block_pose_from_info(info: dict) -> np.ndarray:
    return np.asarray(info["block_pose"], dtype=float).copy()


def agent_pos_from_info(info: dict) -> np.ndarray:
    return np.asarray(info["pos_agent"], dtype=float).copy()


def clip_action(action: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(action, dtype=float), 0.0, 512.0)


def estimate_dubins_limit_surface_from_b(
    b: np.ndarray, c: np.ndarray, rho: float, eps: float = 1e-9
):
    """Estimate scale-invariant Dubins LS coefficients from normalized contact model b."""
    b_xy = b[:2]
    c_xy = c[:2]
    denom_xy = float(np.dot(b_xy, b_xy))
    if denom_xy < eps:
        raise ValueError(
            "Cannot estimate Dubins LS: planar components of b are too small."
        )
    planar_scale = float(np.dot(b_xy, c_xy) / denom_xy)

    if abs(float(b[2])) < eps:
        raise ValueError(
            "Cannot estimate Dubins LS: rotational component b[2] is too small."
        )
    rotational_scale = float(c[2] / b[2])

    # Only ratio a_ls / b_ls is observable from normalized b.
    ratio_ab = abs((planar_scale / rotational_scale) * (rho**2))
    a_ls = ratio_ab
    b_ls = 1.0
    return a_ls, b_ls


def contact_target_from_local(
    block_pose: np.ndarray, pt_local: np.ndarray, local_offset: np.ndarray
):
    return block_pose[:2] + rot2(block_pose[2]) @ (pt_local + local_offset)


def place_pusher_at_local_contact_without_step(
    env,
    info: dict,
    pt_local: np.ndarray,
    recorder: MP4Recorder | None = None,
):
    """Project hidden pusher state onto nominal local contact without moving the block."""
    block_pose = block_pose_from_info(info)
    pusher_xy = clip_action(
        contact_target_from_local(
            block_pose, pt_local, local_offset=np.array([0.0, 0.0], dtype=float)
        )
    )
    env.unwrapped.agent.position = tuple(pusher_xy.tolist())
    env.unwrapped.agent.velocity = (0.0, 0.0)
    info = dict(info)
    info["pos_agent"] = pusher_xy.copy()
    info["vel_agent"] = np.zeros(2, dtype=float)
    if recorder is not None:
        recorder.capture(env)
    return info


def run_sysid_probing(
    env,
    info: dict,
    learner: PushLearner,
    opts_model: OptsModel,
    args,
    pt_action_local: np.ndarray,
    recorder: MP4Recorder | None = None,
):
    """Run alternating local pushes and fit contact model b via SVD."""
    probe_angle = np.deg2rad(args.probe_angle_deg)
    b_est = opts_model.c / np.linalg.norm(opts_model.c)
    flag_last = -1
    accepted = 0

    block_pose = block_pose_from_info(info)
    for i in range(args.n_probe):
        sign = -1.0 if (i % 2) else 1.0
        angle = sign * probe_angle
        local_offset = np.array(
            [
                args.probe_mag * np.cos(angle),
                args.probe_mag * np.sin(angle),
            ],
            dtype=float,
        )
        action = contact_target_from_local(block_pose, pt_action_local, local_offset)
        _, _, terminated, truncated, info_next = env.step(clip_action(action))
        if recorder is not None:
            recorder.capture(env)
        block_pose_next = block_pose_from_info(info_next)

        flag_data = learner.receive_data(block_pose_next, block_pose, opts_model)
        b_curr, flag_last = learner.train_svd(opts_model, flag_data)
        if flag_last > 0:
            b_est = b_curr
            accepted += 1

        block_pose = block_pose_next
        info = info_next
        if terminated or truncated:
            break

    return b_est, flag_last, accepted, info


def rollout_model_with_local_displacements(
    x_start: np.ndarray,
    u_local: np.ndarray,
    T: float,
    D_inv: np.ndarray,
):
    """Roll out planar model f_ with local displacement controls.

    PushPlannerDubin returns per-step local displacements. The dynamics model f_
    expects a speed-like magnitude that is multiplied by T internally, so we
    convert displacement -> speed with mag = ||u_local|| / T.
    """
    n = u_local.shape[1] + 1
    x = np.zeros((3, n), dtype=float)
    x[:, 0] = x_start
    for i in range(n - 1):
        u_i = u_local[:, i]
        mag = float(np.linalg.norm(u_i) / max(float(T), 1e-12))
        ang = float(np.arctan2(u_i[1], u_i[0]))
        x[:, i + 1] = f_(x[:, i], np.array([ang, mag], dtype=float), T, D_inv)
    return x


def pusher_path_from_block_traj(x_traj: np.ndarray, pt_local: np.ndarray):
    n = x_traj.shape[1]
    p = np.zeros((2, n), dtype=float)
    for i in range(n):
        p[:, i] = x_traj[:2, i] + rot2(x_traj[2, i]) @ pt_local
    return p


def execute_open_loop_pusher_targets(
    env,
    info: dict,
    pusher_targets: np.ndarray,
    goal_pose: np.ndarray,
    recorder: MP4Recorder | None = None,
):
    """Execute absolute pusher targets in Push-T and collect true trajectories."""
    n = pusher_targets.shape[1]
    x_true = np.zeros((3, n), dtype=float)
    p_true = np.zeros((2, n), dtype=float)
    a_env_exec = np.zeros((2, max(0, n - 1)), dtype=float)
    coverage_true = np.zeros(n, dtype=float)
    success_true = np.zeros(n, dtype=bool)

    x_true[:, 0] = block_pose_from_info(info)
    p_true[:, 0] = agent_pos_from_info(info)
    goal_geom = tee_geometry_world(goal_pose)
    coverage_true[0] = coverage_from_pose(x_true[:, 0], goal_pose, goal_geom=goal_geom)
    success_threshold = float(env.unwrapped.success_threshold)
    success_true[0] = coverage_true[0] > success_threshold

    end_idx = n - 1
    for i in range(n - 1):
        action = clip_action(pusher_targets[:, i + 1])
        a_env_exec[:, i] = action
        _, _, terminated, truncated, info = env.step(action)
        if recorder is not None:
            recorder.capture(env)
        x_true[:, i + 1] = block_pose_from_info(info)
        p_true[:, i + 1] = agent_pos_from_info(info)
        coverage_i = info.get("coverage", None)
        if coverage_i is None:
            coverage_i = coverage_from_pose(
                x_true[:, i + 1], goal_pose, goal_geom=goal_geom
            )
        coverage_true[i + 1] = float(coverage_i)
        success_true[i + 1] = bool(
            info.get("is_success", coverage_true[i + 1] > success_threshold)
        )
        if terminated or truncated:
            end_idx = i + 1
            break

    return (
        x_true[:, : end_idx + 1],
        p_true[:, : end_idx + 1],
        a_env_exec[:, :end_idx],
        coverage_true[: end_idx + 1],
        success_true[: end_idx + 1],
        info,
    )


def tee_polygons_world(pose: np.ndarray):
    """Return two world polygons for Push-T T-block at the given pose."""
    # Geometry from gym_pusht.envs.pusht.PushTEnv.add_tee with scale=30.
    shape1 = np.array(
        [
            [-60.0, 0.0],
            [60.0, 0.0],
            [60.0, 30.0],
            [-60.0, 30.0],
        ],
        dtype=float,
    )
    shape2 = np.array(
        [
            [-15.0, 30.0],
            [-15.0, 120.0],
            [15.0, 120.0],
            [15.0, 30.0],
        ],
        dtype=float,
    )
    R = rot2(float(pose[2]))
    p = np.asarray(pose[:2], dtype=float)
    poly1 = (R @ shape1.T).T + p
    poly2 = (R @ shape2.T).T + p
    return poly1, poly2


def tee_geometry_world(pose: np.ndarray):
    poly1, poly2 = tee_polygons_world(pose)
    return sg.Polygon(poly1).union(sg.Polygon(poly2))


def coverage_from_pose(block_pose: np.ndarray, goal_pose: np.ndarray, goal_geom=None):
    if goal_geom is None:
        goal_geom = tee_geometry_world(goal_pose)
    block_geom = tee_geometry_world(block_pose)
    goal_area = float(goal_geom.area)
    if goal_area <= 0.0:
        return 0.0
    return float(goal_geom.intersection(block_geom).area / goal_area)


def coverage_from_traj(x_block_traj: np.ndarray, goal_pose: np.ndarray):
    n = x_block_traj.shape[1]
    cov = np.zeros(n, dtype=float)
    goal_geom = tee_geometry_world(goal_pose)
    for i in range(n):
        cov[i] = coverage_from_pose(x_block_traj[:, i], goal_pose, goal_geom=goal_geom)
    return cov


def write_rollout_debug_log(
    log_path: str,
    x_expected: np.ndarray,
    p_expected: np.ndarray,
    u_plan_local: np.ndarray,
    x_true: np.ndarray,
    p_true: np.ndarray,
    p_contact_true_assumed: np.ndarray,
    a_env_exec: np.ndarray,
    coverage_expected: np.ndarray,
    coverage_true: np.ndarray,
    success_threshold: float,
    goal_pose: np.ndarray,
    T: float,
):
    n = min(
        x_expected.shape[1],
        x_true.shape[1],
        coverage_expected.shape[0],
        coverage_true.shape[0],
    )
    pos_err = np.linalg.norm(x_true[:2, :n] - x_expected[:2, :n], axis=0)
    ang_err = np.abs(wrap_to_pi(x_true[2, :n] - x_expected[2, :n]))
    pusher_err = np.linalg.norm(p_true[:, :n] - p_expected[:, :n], axis=0)

    def fmt(v):
        return "nan" if not np.isfinite(v) else f"{float(v):.6f}"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Push-T sys-id Dubins debug log\n")
        f.write(f"# steps={n} dt={T:.6f}\n")
        f.write(
            f"# goal_pose={goal_pose[0]:.6f},{goal_pose[1]:.6f},{goal_pose[2]:.6f}\n"
        )
        f.write(
            "# columns: step | "
            "x_plan y_plan th_plan | p_plan_x p_plan_y | "
            "u_plan_local_x u_plan_local_y u_plan_ang u_plan_mag | "
            "u_env_cmd_x u_env_cmd_y | "
            "x_true y_true th_true | p_true_x p_true_y | "
            "p_contact_true_x p_contact_true_y | "
            "coverage_exp coverage_true success_threshold success_true | "
            "dist_goal_exp dist_goal_true goal_ang_err_exp goal_ang_err_true | "
            "err_pos err_ang err_pusher\n"
        )
        for i in range(n):
            if i < u_plan_local.shape[1]:
                u_lx = u_plan_local[0, i]
                u_ly = u_plan_local[1, i]
                u_ang = np.arctan2(u_ly, u_lx)
                u_mag = np.linalg.norm(u_plan_local[:, i])
            else:
                u_lx, u_ly, u_ang, u_mag = np.nan, np.nan, np.nan, np.nan

            if i < a_env_exec.shape[1]:
                u_env_x = a_env_exec[0, i]
                u_env_y = a_env_exec[1, i]
            else:
                u_env_x, u_env_y = np.nan, np.nan

            row = [
                float(i),
                x_expected[0, i],
                x_expected[1, i],
                x_expected[2, i],
                p_expected[0, i],
                p_expected[1, i],
                u_lx,
                u_ly,
                u_ang,
                u_mag,
                u_env_x,
                u_env_y,
                x_true[0, i],
                x_true[1, i],
                x_true[2, i],
                p_true[0, i],
                p_true[1, i],
                p_contact_true_assumed[0, i],
                p_contact_true_assumed[1, i],
                coverage_expected[i],
                coverage_true[i],
                success_threshold,
                1.0 if coverage_true[i] > success_threshold else 0.0,
                np.linalg.norm(x_expected[:2, i] - goal_pose[:2]),
                np.linalg.norm(x_true[:2, i] - goal_pose[:2]),
                np.abs(wrap_to_pi(x_expected[2, i] - goal_pose[2])),
                np.abs(wrap_to_pi(x_true[2, i] - goal_pose[2])),
                pos_err[i],
                ang_err[i],
                pusher_err[i],
            ]
            f.write(f"{int(row[0])}\t" + "\t".join(fmt(v) for v in row[1:]) + "\n")

    print(f"Saved rollout debug table to {log_path}")


def linspace_with_margin(start: float, stop: float, count: int, margin: float) -> np.ndarray:
    low = float(start + margin)
    high = float(stop - margin)
    if high < low:
        center = 0.5 * (start + stop)
        return np.array([center], dtype=float)
    if count <= 1:
        return np.array([0.5 * (low + high)], dtype=float)
    return np.linspace(low, high, int(count), dtype=float)


def build_side_contact_points(
    points_per_side: int, margin: float, sides: tuple[str, ...]
) -> list[tuple[str, np.ndarray]]:
    points: list[tuple[str, np.ndarray]] = []
    n = max(1, int(points_per_side))

    if "left" in sides:
        for i, y in enumerate(linspace_with_margin(0.0, 30.0, n, margin)):
            points.append((f"left{i+1}", np.array([-75.0, float(y)], dtype=float)))
    if "right" in sides:
        for i, y in enumerate(linspace_with_margin(0.0, 30.0, n, margin)):
            points.append((f"right{i+1}", np.array([75.0, float(y)], dtype=float)))
    if "top" in sides:
        for i, x in enumerate(linspace_with_margin(-15.0, 15.0, n, margin)):
            points.append((f"top{i+1}", np.array([float(x), 135.0], dtype=float)))
    if "bottom" in sides:
        for i, x in enumerate(linspace_with_margin(-60.0, 60.0, n, margin)):
            points.append((f"bottom{i+1}", np.array([float(x), -15.0], dtype=float)))

    return points


def evaluate_contact_candidate(args, pt_local: np.ndarray, gym):
    """Evaluate a contact point quickly and return planning/rollout summary."""
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    try:
        reset_options = None
        if args.reset_to_state is not None:
            reset_options = {
                "reset_to_state": np.asarray(args.reset_to_state, dtype=float)
            }
        _, info = env.reset(seed=args.seed, options=reset_options)
        T = 1.0 / env.unwrapped.control_hz
        goal_pose = np.asarray(info["goal_pose"], dtype=float).copy()

        opts_model = OptsModel()
        opts_model.T = T
        opts_model.rho = float(args.rho)
        opts_model.pt = pt_local.copy()
        opts_model.c = np.array(
            [-pt_local[1] / opts_model.rho, pt_local[0] / opts_model.rho, -1.0],
            dtype=float,
        )
        b0 = opts_model.c / np.linalg.norm(opts_model.c)
        set_contact_model_b(opts_model, b0)

        info = place_pusher_at_local_contact_without_step(env, info, pt_local, recorder=None)

        learner_window = int(args.learner_window)
        if learner_window <= 0:
            learner_window = int(args.n_probe)
        learner_window = max(3, min(learner_window, int(args.n_probe)))
        learner = PushLearner(learner_window, args.probe_discount, opts_model)
        b_est, flag_last, accepted, info = run_sysid_probing(
            env,
            info,
            learner,
            opts_model,
            args,
            pt_action_local=pt_local,
            recorder=None,
        )
        if flag_last < 0:
            b_est = b0
        set_contact_model_b(opts_model, b_est)

        x_start = block_pose_from_info(info)
        info = place_pusher_at_local_contact_without_step(env, info, pt_local, recorder=None)
        a_ls, b_ls = estimate_dubins_limit_surface_from_b(b_est, opts_model.c, opts_model.rho)
        dubins = PushPlannerDubin(a_ls, b_ls, np.linalg.norm(pt_local), args.mu)

        goal_pose_plan = goal_pose.copy()
        goal_pose_plan[2] = unwrap_angle_near_reference(
            float(goal_pose_plan[2]), float(x_start[2])
        )
        dubins_step_size = max(
            float(dubins.radius_turn / 50.0), float(args.min_dubins_step_size)
        )
        x_dubins, _ = dubins.plan(x_start, goal_pose_plan, step_size=dubins_step_size)
        p_dubins = pusher_path_from_block_traj(x_dubins, pt_local)

        if args.max_rollout_steps <= 0:
            n_exec = p_dubins.shape[1]
        else:
            n_exec = min(args.max_rollout_steps + 1, p_dubins.shape[1])

        _, _, _, coverage_true, _, _ = execute_open_loop_pusher_targets(
            env, info, p_dubins[:, :n_exec], goal_pose=goal_pose, recorder=None
        )
        return {
            "pt_local": pt_local.copy(),
            "plan_steps": int(x_dubins.shape[1]),
            "exec_steps": int(max(0, n_exec - 1)),
            "coverage_max": float(np.max(coverage_true)),
            "coverage_final": float(coverage_true[-1]),
            "accepted_updates": int(accepted),
            "flag_last": int(flag_last),
        }
    finally:
        env.close()


def tagged_output_path(path_value: str, tag: str) -> str:
    if path_value == "":
        return ""
    path = Path(path_value)
    return str(path.with_name(f"{path.stem}_{tag}{path.suffix}"))


def run_visualization_for_contact(args, gym, pt_local: np.ndarray, run_tag: str):
    save_plot_path = tagged_output_path(args.save_plot, run_tag)
    save_mp4_path = tagged_output_path(args.save_mp4, run_tag)
    debug_log_path = tagged_output_path(args.debug_log, run_tag)

    need_frame_capture = bool(save_mp4_path or save_plot_path)
    render_mode = "rgb_array" if need_frame_capture else args.render_mode
    if need_frame_capture and args.render_mode != "rgb_array":
        print("Overriding render mode to rgb_array for headless MP4 recording.")

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=render_mode)
    try:
        recorder = MP4Recorder(enabled=need_frame_capture)
        reset_options = None
        if args.reset_to_state is not None:
            reset_options = {
                "reset_to_state": np.asarray(args.reset_to_state, dtype=float)
            }
        _, info = env.reset(seed=args.seed, options=reset_options)
        recorder.capture(env)

        T = 1.0 / env.unwrapped.control_hz
        goal_pose = np.asarray(info["goal_pose"], dtype=float).copy()
        print(f"--- Visualization run: {run_tag} ---")
        print(f"Reset seed: {args.seed}")
        print(f"Using contact point: {pt_local}")
        print(f"Initial block pose: {block_pose_from_info(info)}")
        print(f"Goal block pose   : {goal_pose}")
        print(f"Model timestep T={T:.4f}s")
        print(
            "State space used for planning: privileged low-dimensional [x, y, theta] from env info."
        )
        print(
            f"Evaluation metric: Push-T coverage (success threshold={env.unwrapped.success_threshold:.2f})."
        )

        # Contact model setup (same local point in both frames; only world frame rotates).
        opts_model = OptsModel()
        opts_model.T = T
        opts_model.rho = float(args.rho)
        opts_model.pt = pt_local.copy()
        opts_model.c = np.array(
            [-pt_local[1] / opts_model.rho, pt_local[0] / opts_model.rho, -1.0],
            dtype=float,
        )
        b0 = opts_model.c / np.linalg.norm(opts_model.c)
        set_contact_model_b(opts_model, b0)

        # Match demo_planars assumption: start each phase with pusher at nominal contact.
        info = place_pusher_at_local_contact_without_step(
            env, info, pt_local, recorder=recorder
        )

        # Probing + sys-id (no DDP).
        learner_window = int(args.learner_window)
        if learner_window <= 0:
            learner_window = int(args.n_probe)
        learner_window = max(3, min(learner_window, int(args.n_probe)))
        print(
            f"Probing setup (demo_planars-style alternating pushes): "
            f"n_probe={args.n_probe}, learner_window={learner_window}"
        )
        learner = PushLearner(learner_window, args.probe_discount, opts_model)
        b_est, flag_last, accepted, info = run_sysid_probing(
            env,
            info,
            learner,
            opts_model,
            args,
            pt_action_local=pt_local,
            recorder=recorder,
        )
        if flag_last < 0:
            print(
                "Sys-id did not fully converge; falling back to initial contact model guess."
            )
            b_est = b0
        set_contact_model_b(opts_model, b_est)

        print(f"Estimated b: {b_est}")
        print(f"Accepted SVD updates: {accepted}/{args.n_probe}")

        x_start = block_pose_from_info(info)
        info = place_pusher_at_local_contact_without_step(
            env, info, pt_local, recorder=recorder
        )
        print(f"Planning start block pose (env): {x_start}")
        coverage_start = coverage_from_pose(x_start, goal_pose)
        print(f"Coverage at planning start: {coverage_start:.4f}")

        # Dubins from identified model.
        a_ls, b_ls = estimate_dubins_limit_surface_from_b(
            b_est, opts_model.c, opts_model.rho
        )
        dubins = PushPlannerDubin(a_ls, b_ls, np.linalg.norm(pt_local), args.mu)
        print(
            f"Dubins LS (from sys-id): a={a_ls:.6f}, b={b_ls:.6f}, radius={dubins.radius_turn:.4f}"
        )

        # Plan to the environment goal pose, using the nearest equivalent yaw branch
        # to avoid unnecessary 2*pi turns in Dubins heading.
        goal_pose_plan = goal_pose.copy()
        goal_pose_plan[2] = unwrap_angle_near_reference(
            float(goal_pose_plan[2]), float(x_start[2])
        )
        print(
            f"Dubins planning target pose: [{goal_pose_plan[0]:.3f}, {goal_pose_plan[1]:.3f}, {goal_pose_plan[2]:.3f}] "
            f"(env goal yaw={goal_pose[2]:.3f})"
        )

        dubins_step_size = max(
            float(dubins.radius_turn / 50.0), float(args.min_dubins_step_size)
        )
        print(f"Dubins sampling step size: {dubins_step_size:.4f}px")
        x_dubins, u_plan_local = dubins.plan(
            x_start, goal_pose_plan, step_size=dubins_step_size
        )
        p_dubins = pusher_path_from_block_traj(x_dubins, pt_local)
        x_model_roll = rollout_model_with_local_displacements(
            x_start, u_plan_local, T, opts_model.D_inv
        )
        d_goal_start = float(np.linalg.norm(x_dubins[:2, 0] - goal_pose[:2]))
        plan_goal_pos_err = float(np.linalg.norm(x_dubins[:2, -1] - goal_pose[:2]))
        plan_goal_ang_err = float(np.abs(wrap_to_pi(x_dubins[2, -1] - goal_pose[2])))
        model_goal_pos_err = float(np.linalg.norm(x_model_roll[:2, -1] - goal_pose[:2]))
        model_goal_ang_err = float(
            np.abs(wrap_to_pi(x_model_roll[2, -1] - goal_pose[2]))
        )
        print(
            f"Dubins planned distance-to-goal: start={d_goal_start:.4f}px, end={plan_goal_pos_err:.4f}px"
        )
        print(
            f"Dubins planned final-goal error: pos={plan_goal_pos_err:.4f}px, "
            f"ang={plan_goal_ang_err:.4f}rad"
        )
        print(
            f"Model roll-forward final-goal error (same actions): pos={model_goal_pos_err:.4f}px, "
            f"ang={model_goal_ang_err:.4f}rad"
        )
        if plan_goal_pos_err > 1e-3 or plan_goal_ang_err > 1e-3:
            print(
                "[WARNING] Dubins plan did not terminate exactly at requested goal pose. "
                f"pos_err={plan_goal_pos_err:.6f}, ang_err={plan_goal_ang_err:.6f}"
            )

        # Execute absolute pusher targets open-loop in Push-T.
        if args.max_rollout_steps <= 0:
            n_exec = p_dubins.shape[1]
        else:
            n_exec = min(args.max_rollout_steps + 1, p_dubins.shape[1])
        rollout_frame_start = len(recorder.frames)
        if recorder.enabled:
            recorder.capture(env)  # rollout frame at step 0
        x_true, p_true, a_env_exec, coverage_true, success_true, _ = (
            execute_open_loop_pusher_targets(
                env, info, p_dubins[:, :n_exec], goal_pose=goal_pose, recorder=recorder
            )
        )
        x_expected_exec = x_dubins[:, :n_exec]
        p_expected_exec = p_dubins[:, :n_exec]
        coverage_expected = coverage_from_traj(x_expected_exec, goal_pose)
        u_plan_local_exec = u_plan_local[:, : max(0, n_exec - 1)]
        p_contact_true_assumed = pusher_path_from_block_traj(x_true, pt_local)

        # Align lengths for drift metrics.
        n = min(x_expected_exec.shape[1], x_true.shape[1])
        pos_err = np.linalg.norm(x_true[:2, :n] - x_expected_exec[:2, :n], axis=0)
        ang_err = np.abs(wrap_to_pi(x_true[2, :n] - x_expected_exec[2, :n]))
        pusher_err = np.linalg.norm(p_true[:, :n] - p_expected_exec[:, :n], axis=0)
        coverage_expected_exec = coverage_expected[:n]
        coverage_true_exec = coverage_true[:n]
        success_true_exec = success_true[:n]
        success_threshold = float(env.unwrapped.success_threshold)
        d_goal_true = np.linalg.norm(x_true[:2, :n] - goal_pose[:2, None], axis=0)

        print("Drift summary:")
        print(f"  rollout steps                 : {n - 1}")
        print(f"  mean block position drift (px): {float(np.mean(pos_err)):.3f}")
        print(f"  final block position drift(px): {float(pos_err[-1]):.3f}")
        print(f"  mean block angle drift (rad)  : {float(np.mean(ang_err)):.4f}")
        print(f"  final block angle drift (rad) : {float(ang_err[-1]):.4f}")
        print(f"  mean pusher tracking drift(px): {float(np.mean(pusher_err)):.3f}")
        print(
            "  note: pusher tracking drift = ||actual pusher position - planner contact point||"
        )
        print("Coverage summary (environment metric):")
        print(f"  start coverage                : {float(coverage_true_exec[0]):.4f}")
        print(
            f"  max coverage                  : {float(np.max(coverage_true_exec)):.4f}"
        )
        print(f"  final coverage                : {float(coverage_true_exec[-1]):.4f}")
        print(f"  success threshold             : {success_threshold:.4f}")
        print(f"  any success during rollout    : {bool(np.any(success_true_exec))}")
        print(
            f"  coverage progress (final-start): "
            f"{float(coverage_true_exec[-1] - coverage_true_exec[0]):+.4f}"
        )
        print("Goal-center distance (diagnostic only):")
        print(
            f"  start / min / final (px)      : {float(d_goal_true[0]):.3f} / {float(np.min(d_goal_true)):.3f} / {float(d_goal_true[-1]):.3f}"
        )

        if save_plot_path:
            pusht_start_frame = recorder.frames[rollout_frame_start]
            save_start_comparison_image(
                pusht_start_frame_rgb=pusht_start_frame,
                x_traj=x_expected_exec,
                p_pusher_traj=p_expected_exec,
                goal_pose=goal_pose,
                get_tee_polygons=tee_polygons_world,
                save_path=save_plot_path,
            )
        if debug_log_path:
            write_rollout_debug_log(
                log_path=debug_log_path,
                x_expected=x_expected_exec,
                p_expected=p_expected_exec,
                u_plan_local=u_plan_local_exec,
                x_true=x_true,
                p_true=p_true,
                p_contact_true_assumed=p_contact_true_assumed,
                a_env_exec=a_env_exec,
                coverage_expected=coverage_expected_exec,
                coverage_true=coverage_true_exec,
                success_threshold=success_threshold,
                goal_pose=goal_pose,
                T=T,
            )
        if save_mp4_path:
            mp4_fps = (
                float(args.mp4_fps)
                if args.mp4_fps is not None
                else float(env.unwrapped.metadata["render_fps"])
            )
            rollout_frames = recorder.frames[
                rollout_frame_start : rollout_frame_start + x_true.shape[1]
            ]
            save_frames_to_mp4(rollout_frames, save_mp4_path, mp4_fps)
    finally:
        env.close()


def get_config():
    return SimpleNamespace(
        seed=SEED,
        reset_to_state=RESET_TO_STATE,
        n_probe=N_PROBE,
        probe_angle_deg=PROBE_ANGLE_DEG,
        probe_mag=PROBE_MAG,
        probe_discount=PROBE_DISCOUNT,
        learner_window=LEARNER_WINDOW,
        rho=RHO,
        pt_local=PT_LOCAL.copy(),
        mu=MU,
        max_rollout_steps=MAX_ROLLOUT_STEPS,
        min_dubins_step_size=MIN_DUBINS_STEP_SIZE,
        render_mode=RENDER_MODE,
        save_plot=OUTPUT_PLOT_PATH if SAVE_PLOT else "",
        save_mp4=OUTPUT_ROLLOUT_MP4_PATH if SAVE_ROLLOUT_MP4 else "",
        mp4_fps=OUTPUT_FPS,
        debug_log=OUTPUT_DEBUG_LOG_PATH if SAVE_DEBUG_LOG else "",
        search_contact_points=SEARCH_CONTACT_POINTS,
        contact_points_per_side=CONTACT_POINTS_PER_SIDE,
        contact_corner_margin=CONTACT_CORNER_MARGIN,
        contact_sides=tuple(CONTACT_SIDES),
    )


def main():
    args = get_config()
    ensure_gym_pusht_importable()
    import gym_pusht  # noqa: F401
    import gymnasium as gym

    np.random.seed(args.seed)

    selected_runs: list[tuple[str, np.ndarray]] = []
    if bool(args.search_contact_points):
        candidate_points = build_side_contact_points(
            args.contact_points_per_side,
            args.contact_corner_margin,
            args.contact_sides,
        )
        print(
            "Comprehensive search over contact points along Tee sides "
            f"(excluding corners, margin={args.contact_corner_margin:.1f})..."
        )
        search_results = []
        for label, pt_candidate in candidate_points:
            summary = evaluate_contact_candidate(args, pt_candidate, gym)
            summary["label"] = label
            search_results.append(summary)
            print(
                f"  {label:8s} pt={pt_candidate} "
                f"plan_steps={summary['plan_steps']:4d} "
                f"exec_steps={summary['exec_steps']:4d} "
                f"cov_max={summary['coverage_max']:.4f} "
                f"cov_final={summary['coverage_final']:.4f}"
            )
        search_results = sorted(
            search_results,
            key=lambda item: (
                int(item["plan_steps"]),
                -float(item["coverage_max"]),
                -float(item["coverage_final"]),
            ),
        )
        shortest = search_results[0]
        highest_final = max(
            search_results,
            key=lambda item: (
                float(item["coverage_final"]),
                float(item["coverage_max"]),
                -int(item["plan_steps"]),
            ),
        )
        selected_runs.append(("shortest", np.asarray(shortest["pt_local"], dtype=float).copy()))
        selected_runs.append(
            ("highest_final", np.asarray(highest_final["pt_local"], dtype=float).copy())
        )
        print(
            "Selected shortest-frame candidate: "
            f"{shortest['label']} @ {shortest['pt_local']} "
            f"(plan_steps={shortest['plan_steps']}, cov_final={shortest['coverage_final']:.4f})"
        )
        print(
            "Selected highest-final-coverage candidate: "
            f"{highest_final['label']} @ {highest_final['pt_local']} "
            f"(cov_final={highest_final['coverage_final']:.4f}, plan_steps={highest_final['plan_steps']})"
        )
    else:
        selected_runs.append(("default", np.asarray(args.pt_local, dtype=float).copy()))

    for run_tag, pt_local in selected_runs:
        run_visualization_for_contact(args, gym, pt_local, run_tag)


if __name__ == "__main__":
    main()
