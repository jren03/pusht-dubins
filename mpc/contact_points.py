from __future__ import annotations

import numpy as np


def linspace_with_margin(start: float, stop: float, count: int, margin: float) -> np.ndarray:
    low = float(start + margin)
    high = float(stop - margin)
    if high < low:
        return np.array([0.5 * (start + stop)], dtype=float)
    if count <= 1:
        return np.array([0.5 * (low + high)], dtype=float)
    return np.linspace(low, high, int(count), dtype=float)


def side_contact_points(
    sides: tuple[str, ...], points_per_side: int, corner_margin: float
) -> list[tuple[str, np.ndarray]]:
    n = max(1, int(points_per_side))
    out: list[tuple[str, np.ndarray]] = []
    if "left" in sides:
        for idx, y_val in enumerate(linspace_with_margin(0.0, 30.0, n, corner_margin)):
            out.append((f"left{idx+1}", np.array([-75.0, float(y_val)], dtype=float)))
    if "right" in sides:
        for idx, y_val in enumerate(linspace_with_margin(0.0, 30.0, n, corner_margin)):
            out.append((f"right{idx+1}", np.array([75.0, float(y_val)], dtype=float)))
    if "top" in sides:
        for idx, x_val in enumerate(linspace_with_margin(-15.0, 15.0, n, corner_margin)):
            out.append((f"top{idx+1}", np.array([float(x_val), 135.0], dtype=float)))
    if "bottom" in sides:
        for idx, x_val in enumerate(linspace_with_margin(-60.0, 60.0, n, corner_margin)):
            out.append((f"bottom{idx+1}", np.array([float(x_val), -15.0], dtype=float)))
    return out


def refine_focus_points(
    candidates: list[tuple[str, np.ndarray]],
    focus_labels: tuple[str, ...],
    local_refine_delta: float,
) -> list[tuple[str, np.ndarray]]:
    by_label = {label: point for label, point in candidates}
    refined = list(candidates)
    for label in focus_labels:
        if label not in by_label:
            continue
        center = by_label[label]
        for offset_idx, delta in enumerate([-local_refine_delta, local_refine_delta]):
            pt = center.copy()
            pt[1] = float(pt[1] + delta)
            refined.append((f"{label}_ref{offset_idx+1}", pt))
    return refined

