"""Push-T integration demo: sys-id, Dubins expected rollout, and drift analysis.

This script intentionally excludes DDP and focuses on:
1) Seeded Push-T reset
2) Contact probing + system identification (weighted SVD)
3) Dubins planning from identified contact model
4) Expected model rollout visualization
5) Open-loop execution in Push-T to quantify drift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

from planar_pushing_tools.config import OptsModel, set_contact_model_b
from planar_pushing_tools.model import f_
from planar_pushing_tools.push_learner import PushLearner
from planar_pushing_tools.push_planner_dubin import PushPlannerDubin


class MP4Recorder:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.frames = []

    def capture(self, env):
        if not self.enabled:
            return
        frame = env.render()
        if frame is None:
            raise RuntimeError("Failed to capture frame: env.render() returned None.")
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(
                f"Unexpected frame shape from env.render(): {frame.shape}"
            )
        self.frames.append(frame.copy())

    def save(self, output_path: str, fps: float):
        if not self.enabled:
            return
        if len(self.frames) == 0:
            print("No frames captured; skipping MP4 save.")
            return

        try:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        except ImportError as exc:
            raise ImportError(
                "moviepy is required for MP4 export via ImageSequenceClip. "
                "Install with: pip install moviepy"
            ) from exc

        clip = ImageSequenceClip(self.frames, fps=fps)
        clip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
        clip.close()
        print(f"Saved MP4 to {output_path} ({len(self.frames)} frames @ {fps:.2f} fps)")


def _ensure_gym_pusht_importable():
    repo_root = Path(__file__).resolve().parent
    local_pkg = repo_root / "gym-pusht"
    if str(local_pkg) not in sys.path:
        sys.path.insert(0, str(local_pkg))


def wrap_to_pi(angle: np.ndarray | float):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


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


def run_contact_alignment(
    env,
    info: dict,
    pt_local: np.ndarray,
    n_steps: int,
    inward_offset: float,
    recorder: MP4Recorder | None = None,
):
    block_pose = block_pose_from_info(info)
    local_offset = np.array([inward_offset, 0.0], dtype=float)
    for _ in range(n_steps):
        action = contact_target_from_local(block_pose, pt_local, local_offset)
        _, _, terminated, truncated, info = env.step(clip_action(action))
        if recorder is not None:
            recorder.capture(env)
        block_pose = block_pose_from_info(info)
        if terminated or truncated:
            break
    return info


def run_sysid_probing(
    env,
    info: dict,
    learner: PushLearner,
    opts_model: OptsModel,
    args,
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
        action = contact_target_from_local(block_pose, opts_model.pt, local_offset)
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
    """Roll out planar model f_ with local displacement controls."""
    n = u_local.shape[1] + 1
    x = np.zeros((3, n), dtype=float)
    x[:, 0] = x_start
    for i in range(n - 1):
        u_i = u_local[:, i]
        mag = float(np.linalg.norm(u_i))
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
    recorder: MP4Recorder | None = None,
):
    """Execute absolute pusher targets in Push-T and collect true trajectories."""
    n = pusher_targets.shape[1]
    x_true = np.zeros((3, n), dtype=float)
    p_true = np.zeros((2, n), dtype=float)
    a_env_exec = np.zeros((2, max(0, n - 1)), dtype=float)

    x_true[:, 0] = block_pose_from_info(info)
    p_true[:, 0] = agent_pos_from_info(info)

    end_idx = n - 1
    for i in range(n - 1):
        action = clip_action(pusher_targets[:, i + 1])
        a_env_exec[:, i] = action
        _, _, terminated, truncated, info = env.step(action)
        if recorder is not None:
            recorder.capture(env)
        x_true[:, i + 1] = block_pose_from_info(info)
        p_true[:, i + 1] = agent_pos_from_info(info)
        if terminated or truncated:
            end_idx = i + 1
            break

    return x_true[:, : end_idx + 1], p_true[:, : end_idx + 1], a_env_exec[:, :end_idx], info


def tee_polygons_world(pose: np.ndarray):
    """Return two world polygons for Push-T T-block at the given pose."""
    # Geometry from gym_pusht.envs.pusht.PushTEnv.add_tee with scale=30.
    shape1 = np.array(
        [
            [-60.0, 30.0],
            [60.0, 30.0],
            [60.0, 0.0],
            [-60.0, 0.0],
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
    cog = np.array([0.0, 45.0], dtype=float)
    R = rot2(float(pose[2]))
    p = np.asarray(pose[:2], dtype=float)
    poly1 = (R @ (shape1 - cog).T).T + p
    poly2 = (R @ (shape2 - cog).T).T + p
    return poly1, poly2


def draw_rollout_panel(
    ax,
    x_block: np.ndarray,
    p_pusher: np.ndarray,
    goal_pose: np.ndarray,
    title: str,
    block_color: str,
    pusher_color: str,
):
    ax.plot(x_block[0, :], x_block[1, :], color=block_color, linewidth=2.0, label="Block")
    ax.plot(p_pusher[0, :], p_pusher[1, :], "--", color=pusher_color, linewidth=1.8, label="Pusher")

    start_poly1, start_poly2 = tee_polygons_world(x_block[:, 0])
    goal_poly1, goal_poly2 = tee_polygons_world(goal_pose)
    final_poly1, final_poly2 = tee_polygons_world(x_block[:, -1])
    ax.add_patch(Polygon(start_poly1, closed=True, fill=False, edgecolor="navy", linewidth=1.4))
    ax.add_patch(Polygon(start_poly2, closed=True, fill=False, edgecolor="navy", linewidth=1.4))
    ax.add_patch(Polygon(goal_poly1, closed=True, fill=False, edgecolor="green", linewidth=2.0))
    ax.add_patch(Polygon(goal_poly2, closed=True, fill=False, edgecolor="green", linewidth=2.0))
    ax.add_patch(Polygon(final_poly1, closed=True, fill=False, edgecolor=block_color, linewidth=1.6))
    ax.add_patch(Polygon(final_poly2, closed=True, fill=False, edgecolor=block_color, linewidth=1.6))

    ax.add_patch(Circle(p_pusher[:, 0], radius=15.0, fill=False, edgecolor=pusher_color, linewidth=1.2))
    ax.plot(goal_pose[0], goal_pose[1], "g*", markersize=10, label="Goal center")

    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")


def plot_results(
    x_expected: np.ndarray,
    p_expected: np.ndarray,
    x_true: np.ndarray,
    p_true: np.ndarray,
    goal_pose: np.ndarray,
    save_path: str | None,
    show_plot: bool,
):
    n = min(x_expected.shape[1], x_true.shape[1])
    x_expected = x_expected[:, :n]
    p_expected = p_expected[:, :n]
    x_true = x_true[:, :n]
    p_true = p_true[:, :n]

    pos_err = np.linalg.norm(x_true[:2, :] - x_expected[:2, :], axis=0)
    ang_err = np.abs(wrap_to_pi(x_true[2, :] - x_expected[2, :]))
    pusher_err = np.linalg.norm(p_true - p_expected, axis=0)

    fig, (ax_exp, ax_env, ax_err) = plt.subplots(1, 3, figsize=(19, 6.2), num=0)
    draw_rollout_panel(
        ax=ax_exp,
        x_block=x_expected,
        p_pusher=p_expected,
        goal_pose=goal_pose,
        title="Anticipated rollout (model)",
        block_color="tab:blue",
        pusher_color="tab:cyan",
    )
    draw_rollout_panel(
        ax=ax_env,
        x_block=x_true,
        p_pusher=p_true,
        goal_pose=goal_pose,
        title="Actual rollout (Push-T env)",
        block_color="tab:red",
        pusher_color="tab:pink",
    )

    # Drift
    t = np.arange(n)
    ax_err.plot(t, pos_err, "-r", label="Block position drift (px)")
    ax_err.plot(t, ang_err, "-b", label="Block angle drift (rad)")
    ax_err.plot(t, pusher_err, "-m", label="Pusher tracking drift (px)")
    ax_err.set_title("Drift over rollout")
    ax_err.set_xlabel("step")
    ax_err.set_ylabel("error")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def write_rollout_debug_log(
    log_path: str,
    x_expected: np.ndarray,
    p_expected: np.ndarray,
    u_plan_local: np.ndarray,
    x_true: np.ndarray,
    p_true: np.ndarray,
    a_env_exec: np.ndarray,
    T: float,
):
    n = min(x_expected.shape[1], x_true.shape[1])
    pos_err = np.linalg.norm(x_true[:2, :n] - x_expected[:2, :n], axis=0)
    ang_err = np.abs(wrap_to_pi(x_true[2, :n] - x_expected[2, :n]))
    pusher_err = np.linalg.norm(p_true[:, :n] - p_expected[:, :n], axis=0)

    def fmt(v):
        return "nan" if not np.isfinite(v) else f"{float(v):.6f}"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Push-T sys-id Dubins debug log\n")
        f.write(f"# steps={n} dt={T:.6f}\n")
        f.write(
            "# columns: step | "
            "x_exp y_exp th_exp | p_exp_x p_exp_y | "
            "u_plan_local_x u_plan_local_y u_plan_ang u_plan_mag | "
            "u_env_cmd_x u_env_cmd_y | "
            "x_true y_true th_true | p_true_x p_true_y | "
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
                pos_err[i],
                ang_err[i],
                pusher_err[i],
            ]
            f.write(f"{int(row[0])}\t" + "\t".join(fmt(v) for v in row[1:]) + "\n")

    print(f"Saved rollout debug table to {log_path}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed used in env.reset(seed=...)."
    )
    parser.add_argument(
        "--reset-to-state",
        type=float,
        nargs=5,
        default=None,
        metavar=("AGENT_X", "AGENT_Y", "BLOCK_X", "BLOCK_Y", "BLOCK_THETA"),
        help="Optional explicit Push-T reset state.",
    )
    parser.add_argument(
        "--n-probe", type=int, default=10, help="Number of probing pushes for sys-id."
    )
    parser.add_argument(
        "--probe-angle-deg",
        type=float,
        default=20.0,
        help="Alternating probe angle magnitude.",
    )
    parser.add_argument(
        "--probe-mag",
        type=float,
        default=8.0,
        help="Probe displacement magnitude in px.",
    )
    parser.add_argument(
        "--probe-discount", type=float, default=0.9, help="SVD data discount factor."
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=60.0,
        help="Characteristic length rho for contact model.",
    )
    parser.add_argument(
        "--contact-radius",
        type=float,
        default=75.0,
        help="Distance from block COM to nominal pusher contact point (px).",
    )
    parser.add_argument(
        "--align-steps",
        type=int,
        default=20,
        help="Steps to align pusher to contact before/after probe.",
    )
    parser.add_argument(
        "--align-inward-offset",
        type=float,
        default=2.0,
        help="Extra local +x offset during alignment to keep light preload.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.3,
        help="Friction coefficient used by Dubins model.",
    )
    parser.add_argument(
        "--max-rollout-steps",
        type=int,
        default=160,
        help="Maximum number of planned steps executed in environment.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="rgb_array",
        choices=["rgb_array", "human"],
        help="Push-T render mode.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="pusht_sysid_dubins_rollout.png",
        help="Path to save matplotlib visualization. Use empty string to disable.",
    )
    parser.add_argument(
        "--save-mp4",
        type=str,
        default="pusht_sysid_dubins_rollout.mp4",
        help="Path to save headless rollout video via ImageSequenceClip. Use empty string to disable.",
    )
    parser.add_argument(
        "--mp4-fps",
        type=float,
        default=None,
        help="FPS for saved MP4. Default: env render_fps.",
    )
    parser.add_argument(
        "--debug-log",
        type=str,
        default="pusht_sysid_dubins_rollout_debug.txt",
        help="Path to save per-step anticipated vs actual state/action table. Use empty string to disable.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open matplotlib window (useful for headless runs).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    _ensure_gym_pusht_importable()
    try:
        import gymnasium as gym
        import gym_pusht  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Failed to import gymnasium/gym_pusht. Install local gym-pusht dependencies first."
        ) from exc

    np.random.seed(args.seed)

    render_mode = "rgb_array" if args.save_mp4 else args.render_mode
    if args.save_mp4 and args.render_mode != "rgb_array":
        print("Overriding render mode to rgb_array for headless MP4 recording.")
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=render_mode)
    try:
        recorder = MP4Recorder(enabled=bool(args.save_mp4))
        reset_options = None
        if args.reset_to_state is not None:
            reset_options = {
                "reset_to_state": np.asarray(args.reset_to_state, dtype=float)
            }
        _, info = env.reset(seed=args.seed, options=reset_options)
        recorder.capture(env)

        T = 1.0 / env.unwrapped.control_hz
        goal_pose = np.asarray(info["goal_pose"], dtype=float).copy()
        print(f"Reset seed: {args.seed}")
        print(f"Initial block pose: {block_pose_from_info(info)}")
        print(f"Goal block pose   : {goal_pose}")
        print(f"Model timestep T={T:.4f}s")

        # Contact model setup (left-side nominal contact in local frame).
        pt_local = np.array([-args.contact_radius, 0.0], dtype=float)
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

        # Bring pusher near nominal contact before probing.
        info = run_contact_alignment(
            env,
            info,
            pt_local,
            args.align_steps,
            args.align_inward_offset,
            recorder=recorder,
        )

        # Probing + sys-id (no DDP).
        learner = PushLearner(args.n_probe, args.probe_discount, opts_model)
        b_est, flag_last, accepted, info = run_sysid_probing(
            env, info, learner, opts_model, args, recorder=recorder
        )
        if flag_last < 0:
            print(
                "Sys-id did not fully converge; falling back to initial contact model guess."
            )
            b_est = b0
        set_contact_model_b(opts_model, b_est)

        print(f"Estimated b: {b_est}")
        print(f"Accepted SVD updates: {accepted}/{args.n_probe}")

        # Re-align pusher at contact and freeze start pose for planning/rollout.
        info = run_contact_alignment(
            env,
            info,
            pt_local,
            args.align_steps,
            args.align_inward_offset,
            recorder=recorder,
        )
        x_start = block_pose_from_info(info)
        print(f"Planning start block pose: {x_start}")

        # Dubins from identified model.
        a_ls, b_ls = estimate_dubins_limit_surface_from_b(
            b_est, opts_model.c, opts_model.rho
        )
        dubins = PushPlannerDubin(a_ls, b_ls, np.linalg.norm(pt_local), args.mu)
        print(
            f"Dubins LS (from sys-id): a={a_ls:.6f}, b={b_ls:.6f}, radius={dubins.radius_turn:.4f}"
        )

        _, u_plan_local = dubins.plan(x_start, goal_pose)
        x_expected = rollout_model_with_local_displacements(
            x_start, u_plan_local, T, opts_model.D_inv
        )
        p_expected = pusher_path_from_block_traj(x_expected, pt_local)

        # Execute absolute pusher targets open-loop in Push-T.
        n_exec = min(args.max_rollout_steps + 1, p_expected.shape[1])
        x_true, p_true, a_env_exec, _ = execute_open_loop_pusher_targets(
            env, info, p_expected[:, :n_exec], recorder=recorder
        )
        x_expected_exec = x_expected[:, :n_exec]
        p_expected_exec = p_expected[:, :n_exec]
        u_plan_local_exec = u_plan_local[:, : max(0, n_exec - 1)]

        # Align lengths for drift metrics.
        n = min(x_expected_exec.shape[1], x_true.shape[1])
        pos_err = np.linalg.norm(x_true[:2, :n] - x_expected_exec[:2, :n], axis=0)
        ang_err = np.abs(wrap_to_pi(x_true[2, :n] - x_expected_exec[2, :n]))
        pusher_err = np.linalg.norm(p_true[:, :n] - p_expected_exec[:, :n], axis=0)

        print("Drift summary:")
        print(f"  rollout steps                 : {n - 1}")
        print(f"  mean block position drift (px): {float(np.mean(pos_err)):.3f}")
        print(f"  final block position drift(px): {float(pos_err[-1]):.3f}")
        print(f"  mean block angle drift (rad)  : {float(np.mean(ang_err)):.4f}")
        print(f"  final block angle drift (rad) : {float(ang_err[-1]):.4f}")
        print(f"  mean pusher tracking drift(px): {float(np.mean(pusher_err)):.3f}")

        save_path = args.save_plot if args.save_plot else None
        plot_results(
            x_expected=x_expected_exec,
            p_expected=p_expected_exec,
            x_true=x_true,
            p_true=p_true,
            goal_pose=goal_pose,
            save_path=save_path,
            show_plot=not args.no_show,
        )
        if args.debug_log:
            write_rollout_debug_log(
                log_path=args.debug_log,
                x_expected=x_expected_exec,
                p_expected=p_expected_exec,
                u_plan_local=u_plan_local_exec,
                x_true=x_true,
                p_true=p_true,
                a_env_exec=a_env_exec,
                T=T,
            )
        if args.save_mp4:
            mp4_fps = (
                float(args.mp4_fps)
                if args.mp4_fps is not None
                else float(env.unwrapped.metadata["render_fps"])
            )
            recorder.save(args.save_mp4, mp4_fps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
