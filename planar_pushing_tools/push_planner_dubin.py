"""PushPlannerDubin: trajectory planning via Dubins curves in flat output space.

Port of PlanarManipulationToolBox/differential_flat/GetDubinPath.m, adapted
to the convention where the pushing point is at (-r, 0) in the object local
frame and the pushing direction is +x (instead of (0, -r) and +y).
"""

import numpy as np
from .dubins import dubins_init, dubins_path_sample_many, dubins_path_length


class PushPlannerDubin:
    def __init__(self, a, b, r, mu):
        """
        Args:
            a: limit surface diagonal coefficient A[0,0] (assumes A ~ diag(a,a,b_norm))
            b: A[2,2] / rho^2
            r: distance from pushing point to object COM
            mu: friction coefficient at the contact point
        """
        self.a = a
        self.b = b
        self.r = r
        self.mu = mu
        self.radius_turn = a / (b * r * mu)

        # Results
        self.x = None
        self.y = None
        self.theta = None
        self.z = None
        self.u = None
        self.xnom = None

    def _pose_to_flat(self, pose):
        """Map object pose to flat output space.

        Our convention: pushing point at (-r, 0), pushing direction +x.
        In world frame, pushing direction = R @ [1,0] = [cos θ, sin θ].

        Flat output:
            z[0:2] = pos + a/(b*r) * [cos θ, sin θ]
            z[2]   = θ   (Dubins heading equals object heading)
        """
        z = np.zeros(3)
        theta = pose[2]
        offset = self.a / (self.b * self.r)
        z[0] = pose[0] + offset * np.cos(theta)
        z[1] = pose[1] + offset * np.sin(theta)
        z[2] = theta
        return z

    def _flat_to_pose(self, dt, z_xy):
        """Map flat output trajectory back to object poses.

        Inverse mapping for our convention:
            θ = atan2(dz_y, dz_x)
            x = z_x - a/(b*r) * cos θ
            y = z_y - a/(b*r) * sin θ

        Args:
            dt: time step between flat output samples
            z_xy: (2, N) flat output positions

        Returns:
            x, y, theta: arrays of length N-1
        """
        dz = np.diff(z_xy, axis=1) / dt
        theta = np.arctan2(dz[1, :], dz[0, :])
        offset = self.a / (self.b * self.r)
        x = z_xy[0, :-1] - offset * np.cos(theta)
        y = z_xy[1, :-1] - offset * np.sin(theta)
        return x, y, theta

    def plan(self, pose_start, pose_end, step_size=None):
        """Plan a Dubins path from pose_start to pose_end.

        Args:
            pose_start: (3,) start pose [x, y, theta]
            pose_end: (3,) goal pose [x, y, theta]
            step_size: sampling step along the Dubins path
                       (default: radius_turn / 50)

        Returns:
            xnom: (3, N) state trajectory [x; y; theta]
            u: (2, N-1) local-frame Cartesian displacements of pusher
        """
        if step_size is None:
            step_size = self.radius_turn / 50

        # Map to flat output space
        z_start = self._pose_to_flat(pose_start)
        z_end = self._pose_to_flat(pose_end)

        # Solve Dubins path in flat space
        path = dubins_init(z_start, z_end, self.radius_turn)
        if path is None:
            raise RuntimeError("No Dubins path found between configurations")

        # Sample the path
        z_sampled = dubins_path_sample_many(path, step_size)  # (3, M)
        self.z = z_sampled

        # Map flat output back to object poses
        num_z = z_sampled.shape[1]
        T_total = 1.0
        dt = T_total / num_z
        x, y, theta = self._flat_to_pose(dt, z_sampled[0:2, :])

        # Append goal pose as the final point
        x = np.append(x, pose_end[0])
        y = np.append(y, pose_end[1])
        theta = np.append(theta, pose_end[2])

        self.x = x
        self.y = y
        self.theta = theta

        num_pts = len(x)

        # Compute hand (pusher) trajectory in world frame
        hand_local_pt = np.array([-self.r, 0.0])
        hand_q_traj = np.zeros((2, num_pts))
        for i in range(num_pts):
            ct, st = np.cos(theta[i]), np.sin(theta[i])
            R = np.array([[ct, -st], [st, ct]])
            hand_q_traj[:, i] = R @ hand_local_pt + np.array([x[i], y[i]])

        # Compute control: local-frame displacement of the pusher
        u = np.zeros((2, num_pts - 1))
        for i in range(num_pts - 1):
            ct, st = np.cos(theta[i]), np.sin(theta[i])
            R = np.array([[ct, -st], [st, ct]])
            u[:, i] = R.T @ (hand_q_traj[:, i + 1] - hand_q_traj[:, i])

        self.u = u
        self.xnom = np.vstack([x, y, theta])  # (3, num_pts)

        return self.xnom, self.u
