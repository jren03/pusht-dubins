"""PushAnimation: matplotlib-based visualization. Port of wrapper/PushAnimation.m."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FFMpegWriter


class PushAnimation:
    def __init__(self, fig_num, color, pt, W, L, plot_range, x0, xstar,
                 save_video=False, video_path="pushing_demo.mp4", fps=10):
        """
        Args:
            fig_num: figure number
            color: object edge color (r, g, b) tuple, 0-1
            pt: pushing point offset (2,)
            W: object width
            L: object length
            plot_range: [xmin, xmax, ymin, ymax]
            x0: initial state (3,)
            xstar: goal state (3,)
            save_video: if True, record frames to an mp4 file
            video_path: output mp4 file path
            fps: frames per second for the video
        """
        self.pt = pt
        self.W = W
        self.L = L
        self.plot_range = plot_range
        self.color = color

        self.fig, self.ax = plt.subplots(1, 1, num=fig_num)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(plot_range[0], plot_range[1])
        self.ax.set_ylim(plot_range[2], plot_range[3])

        # Video recording
        self.save_video = save_video
        self.writer = None
        if save_video:
            self.writer = FFMpegWriter(fps=fps)
            self.writer.setup(self.fig, video_path, dpi=150)

        red = np.array([222, 45, 38]) / 255.0

        # Goal state rectangle
        self._draw_rect(xstar, red, alpha_edge=0.5, alpha_face=0.2, lw=2)

        # Initial state rectangle
        self._draw_rect(x0, color, alpha_edge=1.0, alpha_face=0.2, lw=2)

        # Current object rectangle (updated each frame)
        self.obj_patch = patches.Rectangle(
            (-W / 2, -L / 2), W, L,
            linewidth=1.5, edgecolor=color, facecolor="none"
        )
        self.ax.add_patch(self.obj_patch)

        # Pusher dot
        self.pusher_dot, = self.ax.plot([], [], "r.", markersize=10)

        # Pusher arrow
        self.pusher_arrow = None

        # Disturbance arrow
        self.dist_arrow = None

        plt.ion()
        plt.show()

    def _draw_rect(self, state, color, alpha_edge=1.0, alpha_face=0.0, lw=1.5):
        """Draw a static rectangle at given pose."""
        x, y, theta = state[0], state[1], state[2]
        rect = patches.Rectangle(
            (-self.W / 2, -self.L / 2), self.W, self.L,
            linewidth=lw, edgecolor=color, facecolor=color,
            alpha=alpha_face,
        )
        t = Affine2D().rotate(theta).translate(x, y) + self.ax.transData
        rect.set_transform(t)
        # Draw edge separately for independent alpha
        rect_edge = patches.Rectangle(
            (-self.W / 2, -self.L / 2), self.W, self.L,
            linewidth=lw, edgecolor=(*color[:3], alpha_edge), facecolor="none",
        )
        rect_edge.set_transform(t)
        self.ax.add_patch(rect)
        self.ax.add_patch(rect_edge)

    def draw_frame(self, X, U, k, dist_mag=None):
        """Update the animation for one frame.

        Args:
            X: current state (3,)
            U: current control (2,)
            k: timestep index
            dist_mag: optional disturbance vector (3,)
        """
        x, y, theta = X[0], X[1], X[2]

        # Update object rectangle
        t = Affine2D().rotate(theta).translate(x, y) + self.ax.transData
        self.obj_patch.set_transform(t)

        # Pusher position (in world frame)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        px = x + cos_t * self.pt[0] - sin_t * self.pt[1]
        py = y + sin_t * self.pt[0] + cos_t * self.pt[1]
        self.pusher_dot.set_data([px], [py])

        # Pusher force direction
        vp_x = U[1] * np.cos(U[0])
        vp_y = U[1] * np.sin(U[0])
        # Transform to world frame
        fx_world = cos_t * vp_x - sin_t * vp_y
        fy_world = sin_t * vp_x + cos_t * vp_y

        if self.pusher_arrow is not None:
            self.pusher_arrow.remove()
        scale = 0.5  # Visual scaling
        self.pusher_arrow = self.ax.annotate(
            "", xy=(px + fx_world * scale, py + fy_world * scale),
            xytext=(px, py),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
        )

        # Disturbance arrow
        if self.dist_arrow is not None:
            self.dist_arrow.remove()
            self.dist_arrow = None
        if dist_mag is not None and np.linalg.norm(dist_mag[:2]) > 1e-6:
            self.dist_arrow = self.ax.annotate(
                "", xy=(x + dist_mag[0] * 3, y + dist_mag[1] * 3),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="red", lw=3),
            )

        self.fig.canvas.draw_idle()
        if self.save_video and self.writer is not None:
            self.writer.grab_frame()
        plt.pause(0.01)

    def finish_video(self):
        """Finalize and save the mp4 video file."""
        if self.writer is not None:
            self.writer.finish()
            self.writer = None
