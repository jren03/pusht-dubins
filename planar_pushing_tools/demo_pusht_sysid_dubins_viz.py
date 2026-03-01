"""Visualization helpers for Push-T sys-id Dubins demo: MP4 recording and start-comparison image."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon


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
        save_frames_to_mp4(self.frames, output_path, fps)
        print(f"Saved MP4 to {output_path} ({len(self.frames)} frames @ {fps:.2f} fps)")


def save_frames_to_mp4(frames: list[np.ndarray], output_path: str, fps: float):
    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    except ImportError as exc:
        raise ImportError(
            "moviepy is required for MP4 export via ImageSequenceClip. "
            "Install with: pip install moviepy"
        ) from exc
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec="libx264", audio=False, logger=None)
    clip.close()


def save_start_comparison_image(
    pusht_start_frame_rgb: np.ndarray,
    x_traj: np.ndarray,
    p_pusher_traj: np.ndarray,
    goal_pose: np.ndarray,
    get_tee_polygons,
    save_path: str,
):
    """Save a single side-by-side image: Push-T starting state (left) | matplotlib start + full trajectory (right)."""
    h, w = pusht_start_frame_rgb.shape[:2]
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    ax_left.imshow(pusht_start_frame_rgb)
    ax_left.set_title("Push-T starting state")
    ax_left.set_xlim(0, w - 1)
    ax_left.set_ylim(h - 1, 0)
    ax_left.set_aspect("equal")

    # Right: start pose + full planned trajectory
    start_poly1, start_poly2 = get_tee_polygons(x_traj[:, 0])
    goal_poly1, goal_poly2 = get_tee_polygons(goal_pose)
    ax_right.plot(
        x_traj[0, :], x_traj[1, :], color="tab:blue", linewidth=2.0, label="Block"
    )
    ax_right.plot(
        p_pusher_traj[0, :],
        p_pusher_traj[1, :],
        "--",
        color="tab:cyan",
        linewidth=1.8,
        label="Pusher",
    )
    ax_right.add_patch(
        Polygon(start_poly1, closed=True, fill=False, edgecolor="navy", linewidth=1.4)
    )
    ax_right.add_patch(
        Polygon(start_poly2, closed=True, fill=False, edgecolor="navy", linewidth=1.4)
    )
    ax_right.add_patch(
        Polygon(goal_poly1, closed=True, fill=False, edgecolor="green", linewidth=2.0)
    )
    ax_right.add_patch(
        Polygon(goal_poly2, closed=True, fill=False, edgecolor="green", linewidth=2.0)
    )
    ax_right.add_patch(
        Circle(
            p_pusher_traj[:, 0],
            radius=15.0,
            fill=False,
            edgecolor="tab:cyan",
            linewidth=1.2,
        )
    )
    ax_right.plot(goal_pose[0], goal_pose[1], "g*", markersize=10, label="Goal center")
    ax_right.set_title("Planned start + full trajectory (matplotlib)")
    ax_right.set_xlabel("x (px)")
    ax_right.set_ylabel("y (px)")
    ax_right.set_xlim(0, 512)
    ax_right.set_ylim(512, 0)
    ax_right.set_aspect("equal")
    ax_right.grid(True, alpha=0.25)
    ax_right.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved start comparison image to {save_path}")
