from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProbeConfig:
    n_probe: int = 40
    probe_angle_deg: float = 45.0
    probe_mag: float = 12.0
    probe_discount: float = 0.9
    learner_window: int = 12


@dataclass
class OnlineUpdateConfig:
    enabled: bool = False
    refresh_every_steps: int = 10
    min_delta_xy: float = 0.5


@dataclass
class ReplanConfig:
    horizon_steps: int = 30
    execute_chunk_steps: int = 3
    min_dubins_step_size: float = 1.2
    min_projected_progress_px: float = 1.0
    shortlist_length_margin_steps: int = 2
    min_empirical_gain_px: float = 1.0
    empirical_gain_window_steps: int = 20
    stagnation_window_steps: int = 40
    stagnation_min_delta_coverage: float = 0.01
    force_contact_switch_on_stagnation: bool = True
    contact_switch_cooldown_steps: int = 20
    action_boundary_margin_px: float = 8.0
    reacquire_steps_after_switch: int = 2
    reacquire_alpha: float = 0.65
    reacquire_contact_target_n: int = 1
    replan_on_contact_loss_streak: int = 2
    replan_on_clipped_action: bool = True
    replan_on_tracking_err_px: float = 20.0


@dataclass
class CostWeights:
    coverage_weight: float = 1.0
    clipping_penalty: float = 0.2
    contact_loss_penalty: float = 0.15
    model_mismatch_penalty: float = 0.15
    replan_penalty: float = 0.05


@dataclass
class ClosedLoopRunConfig:
    seed: int = 1
    reset_to_state: list[float] | None = None
    render_mode: str = "rgb_array"
    rho: float = 60.0
    mu: float = 0.2
    max_rollout_steps: int = 180
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    replan: ReplanConfig = field(default_factory=ReplanConfig)
    online_update: OnlineUpdateConfig = field(default_factory=OnlineUpdateConfig)
    save_plot_path: str = "pusht_closed_loop_start.png"
    save_mp4_path: str = "pusht_closed_loop_rollout.mp4"
    save_debug_log_path: str = "pusht_closed_loop_debug.jsonl"
    save_summary_path: str = "pusht_closed_loop_summary.json"
    mp4_fps: float | None = None


@dataclass
class ContactSearchConfig:
    sides: tuple[str, ...] = ("left", "right", "top", "bottom")
    points_per_side: int = 7
    corner_margin: float = 5.0
    focus_labels: tuple[str, ...] = ("left5", "left6", "left7")
    local_refine_delta: float = 2.5
    include_refinements: bool = True


@dataclass
class SweepConfig:
    output_dir: Path
    seeds: list[int] = field(default_factory=lambda: [1, 2])
    horizon_steps: list[int] = field(default_factory=lambda: [20, 30, 40])
    execute_chunk_steps: list[int] = field(default_factory=lambda: [1, 3, 5])
    min_dubins_step_size: list[float] = field(default_factory=lambda: [1.2, 1.5, 1.8])
    mu_values: list[float] = field(default_factory=lambda: [0.15, 0.2, 0.25])
    rho_values: list[float] = field(default_factory=lambda: [45.0, 60.0, 75.0])
    online_update_every_steps: list[int] = field(default_factory=lambda: [0, 10, 20])
    parallel_workers: int = 8

