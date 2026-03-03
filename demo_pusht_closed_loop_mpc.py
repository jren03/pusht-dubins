"""Closed-loop Push-T receding-horizon Dubins demo with optional sweep."""

from __future__ import annotations

import copy
from pathlib import Path

from mpc.config import ClosedLoopRunConfig, ContactSearchConfig, CostWeights, SweepConfig
from mpc.controller import evaluate_contact_point, run_closed_loop_episode
from mpc.logging import write_json
from mpc.sweep import build_contact_candidates, run_compact_sweep


RUN_SWEEP = False
SWEEP_OUTPUT_DIR = Path("diagnosis_outputs/closed_loop_sweep")


def best_shortest_candidate(rows: list[dict]) -> dict:
    ranked = sorted(rows, key=lambda row: (-float(row["coverage_final"]), -float(row["coverage_max"])))
    top = ranked[: max(1, min(10, len(ranked)))]
    return sorted(top, key=lambda row: int(row["steps"]))[0]


def main() -> None:
    base_cfg = ClosedLoopRunConfig(
        seed=1,
        rho=60.0,
        mu=0.2,
        max_rollout_steps=180,
        save_plot_path="pusht_closed_loop_start.png",
        save_mp4_path="pusht_closed_loop_rollout.mp4",
        save_debug_log_path="pusht_closed_loop_debug.jsonl",
        save_summary_path="pusht_closed_loop_summary.json",
    )
    contact_cfg = ContactSearchConfig(
        sides=("left", "right", "top", "bottom"),
        points_per_side=7,
        corner_margin=5.0,
        focus_labels=("left5", "left6", "left7"),
        local_refine_delta=2.5,
        include_refinements=True,
    )

    if RUN_SWEEP:
        sweep_cfg = SweepConfig(output_dir=SWEEP_OUTPUT_DIR)
        weights = CostWeights()
        summary = run_compact_sweep(base_cfg, contact_cfg, sweep_cfg, weights)
        print("Closed-loop sweep completed.")
        print(f"  runs_total={summary['runs_total']}")
        print(f"  runs_successful={summary['runs_successful']}")
        print(f"  runs_failed={summary['runs_failed']}")
        print(f"  mean_coverage_final={summary['mean_coverage_final']:.4f}")
        print(f"  mean_coverage_max={summary['mean_coverage_max']:.4f}")
        return

    print("Evaluating contact candidates for closed-loop selection...")
    contact_candidates = build_contact_candidates(contact_cfg)
    eval_rows = []
    for label, pt_local in contact_candidates:
        row = evaluate_contact_point(base_cfg, pt_local=to_np(pt_local), label=label)
        eval_rows.append(row)
        print(
            f"  {label:12s} cov_final={row['coverage_final']:.4f} "
            f"cov_max={row['coverage_max']:.4f} steps={row['steps']:3d}"
        )

    write_json("pusht_closed_loop_contact_eval.json", {"rows": eval_rows})
    best_final = max(
        eval_rows,
        key=lambda row: (
            float(row["coverage_final"]),
            float(row["coverage_max"]),
            -int(row["steps"]),
        ),
    )
    best_short = best_shortest_candidate(eval_rows)

    print(
        "Selected highest-final candidate: "
        f"{best_final['label']} @ {best_final['pt_local']} "
        f"(final={best_final['coverage_final']:.4f}, steps={best_final['steps']})"
    )
    print(
        "Selected shortest candidate among strong performers: "
        f"{best_short['label']} @ {best_short['pt_local']} "
        f"(final={best_short['coverage_final']:.4f}, steps={best_short['steps']})"
    )

    best_final_cfg = copy.deepcopy(base_cfg)
    best_final_cfg.save_plot_path = "pusht_closed_loop_start_highest_final.png"
    best_final_cfg.save_mp4_path = "pusht_closed_loop_rollout_highest_final.mp4"
    best_final_cfg.save_debug_log_path = "pusht_closed_loop_debug_highest_final.jsonl"
    best_final_cfg.save_summary_path = "pusht_closed_loop_summary_highest_final.json"
    run_closed_loop_episode(
        cfg=best_final_cfg,
        pt_local=to_np(best_final["pt_local"]),
        run_tag="highest_final",
        save_artifacts=True,
        contact_candidates=[(label, to_np(values)) for label, values in contact_candidates],
    )

    best_short_cfg = copy.deepcopy(base_cfg)
    best_short_cfg.save_plot_path = "pusht_closed_loop_start_shortest.png"
    best_short_cfg.save_mp4_path = "pusht_closed_loop_rollout_shortest.mp4"
    best_short_cfg.save_debug_log_path = "pusht_closed_loop_debug_shortest.jsonl"
    best_short_cfg.save_summary_path = "pusht_closed_loop_summary_shortest.json"
    run_closed_loop_episode(
        cfg=best_short_cfg,
        pt_local=to_np(best_short["pt_local"]),
        run_tag="shortest",
        save_artifacts=True,
        contact_candidates=[(label, to_np(values)) for label, values in contact_candidates],
    )


def to_np(values):
    import numpy as np

    return np.asarray(values, dtype=float)


if __name__ == "__main__":
    main()

