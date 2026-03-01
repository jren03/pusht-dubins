"""PushLearner: online contact model identification via SVD. Port of wrapper/PushLearner.m."""

import numpy as np


class PushLearner:
    def __init__(self, data_NH, discount, opts_model):
        """
        Args:
            data_NH: learning horizon (number of past observations to keep)
            discount: exponential discount factor for weighting
            opts_model: OptsModel instance
        """
        self.data = np.zeros((3, data_NH))
        self.min_rejection_dist = 0.001
        self.Y = None  # Most recent twist observation

        # SVD weights: discount^[1, 2, ..., data_NH], normalized
        self.weight = discount ** np.arange(1, data_NH + 1, dtype=float)
        self.weight = self.weight / np.sum(self.weight)

    def receive_data(self, x, xold, opts_model):
        """Transform state transition to local twist coordinates.

        Args:
            x: new state (3,)
            xold: previous state (3,)
            opts_model: OptsModel instance

        Returns:
            flag: 1 if data accepted, -1 if motion too small
        """
        dx = x - xold

        if np.linalg.norm(dx[:2]) < self.min_rejection_dist:
            print("[WARNING] PushLearner: motion too small")
            return -1

        theta = xold[2]
        Rot = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,            0.0,           1.0],
        ])
        self.Y = np.diag([1.0, 1.0, opts_model.rho]) @ np.linalg.inv(Rot) @ dx
        return 1

    def train_svd(self, opts_model, flag):
        """Estimate contact model b via weighted SVD.

        Args:
            opts_model: OptsModel instance
            flag: from receive_data (-1 means skip)

        Returns:
            b: estimated contact normal (3,) or 0 if failed
            flag: 1 if success, -1 if failed
        """
        if flag < 0:
            return 0, flag

        # Shift data buffer and insert new observation
        print("New data accepted!")
        self.data[:, 1:] = self.data[:, :-1]
        self.data[:, 0] = self.Y

        # Check if buffer has enough data
        if np.linalg.norm(self.data[:, -1]) < 1e-9:
            return 0, -1

        # Weighted SVD: b'Y = 0
        data_weighted = self.data * self.weight[np.newaxis, :]
        _, S, Vt = np.linalg.svd(data_weighted.T, full_matrices=True)
        b = Vt[-1, :]  # Last right singular vector (null space)

        singular = S
        if abs(singular[-1] - singular[-2]) < 1e-6:
            print("SVD: singularity warning.")
            return 0, -1

        b = b / np.linalg.norm(b)

        # Orient b to match sign of c
        if b @ opts_model.c < 0:
            b = -b

        return b, 1
