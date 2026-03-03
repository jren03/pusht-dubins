from __future__ import annotations

import concurrent.futures
import itertools
import copy
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from mpc.config import ClosedLoopRunConfig, ContactSearchConfig, CostWeights, SweepConfig
from mpc.contact_points import refine_focus_points, side_contact_points
from mpc.controller import run_closed_loop_episode
from mpc.logging import score_summary, write_json


def build_contact_candidates(cfg: ContactSearchConfig) -> list[tuple[str, list[float]]]:
    base = side_contact_points(cfg.sides, cfg.points_per_side, cfg.corner_margin)
    if cfg.include_refinements:
        base = refine_focus_points(base, cfg.focus_labels, cfg.local_refine_delta)
    return [(label, [float(point[0]), float(point[1])]) for label, point in base]


def run_compact_sweep(
    base_cfg: ClosedLoopRunConfig,
    contact_cfg: ContactSearchConfig,
    sweep_cfg: SweepConfig,
    weights: CostWeights,
) -> dict:
    sweep_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    contacts = build_contact_candidates(contact_cfg)
    contact_candidates = [(label, values) for label, values in contacts]

    cases = list(
        itertools.product(
            sweep_cfg.seeds,
            sweep_cfg.horizon_steps,
            sweep_cfg.execute_chunk_steps,
            sweep_cfg.min_dubins_step_size,
            sweep_cfg.mu_values,
            sweep_cfg.rho_values,
            sweep_cfg.online_update_every_steps,
        )
    )

    results: list[dict] = []
    payloads = [
        (
            idx,
            case,
            base_cfg,
            contact_candidates,
            asdict(weights),
            str(sweep_cfg.output_dir),
        )
        for idx, case in enumerate(cases)
    ]

    workers = max(1, int(sweep_cfg.parallel_workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_payload = {pool.submit(run_single_case, payload): payload for payload in payloads}
        for future in tqdm(
            concurrent.futures.as_completed(future_to_payload),
            total=len(future_to_payload),
            desc="Closed-loop sweep",
            unit="case",
        ):
            payload = future_to_payload[future]
            err = future.exception()
            if err is None:
                results.append(future.result())
                continue
            failed_summary = failure_summary_from_payload(payload, err)
            case_dir = build_case_dir(payload)
            write_json(str(case_dir / "summary_failed.json"), failed_summary)
            results.append(failed_summary)

    ranked = sorted(results, key=lambda item: float(item["score"]), reverse=True)
    success_rows = [item for item in results if item["status"] == "ok"]
    failed_rows = [item for item in results if item["status"] == "error"]
    sweep_summary = {
        "runs_total": len(results),
        "runs_successful": len(success_rows),
        "runs_failed": len(failed_rows),
        "top10": ranked[:10],
        "best_successful": ranked_successes(success_rows),
        "mean_coverage_final": float(
            sum(item["coverage_final"] for item in success_rows) / max(1, len(success_rows))
        ),
        "mean_coverage_max": float(
            sum(item["coverage_max"] for item in success_rows) / max(1, len(success_rows))
        ),
        "failure_examples": [
            {
                "idx": item["idx"],
                "error_type": item["error_type"],
                "error_message": item["error_message"],
            }
            for item in failed_rows[:10]
        ],
    }
    write_json(str(Path(sweep_cfg.output_dir) / "sweep_summary.json"), sweep_summary)
    return sweep_summary


def np_array(values: list[float]):
    import numpy as np

    return np.asarray(values, dtype=float)


def run_single_case(payload: tuple) -> dict:
    idx, case, base_cfg, contact_candidates, weights_dict, output_dir = payload
    seed, horizon, execute_chunk, step_size, mu, rho, online_every = case
    init_label, init_pt = contact_candidates[0]

    cfg = copy.deepcopy(base_cfg)
    cfg.seed = int(seed)
    cfg.mu = float(mu)
    cfg.rho = float(rho)
    cfg.replan.horizon_steps = int(horizon)
    cfg.replan.execute_chunk_steps = int(execute_chunk)
    cfg.replan.min_dubins_step_size = float(step_size)
    cfg.online_update.enabled = int(online_every) > 0
    cfg.online_update.refresh_every_steps = int(max(1, online_every))

    case_dir = build_case_dir(payload)
    cfg.save_mp4_path = ""
    cfg.save_plot_path = ""
    cfg.save_debug_log_path = str(case_dir / "steps.jsonl")
    cfg.save_summary_path = str(case_dir / "summary.json")

    episode = run_closed_loop_episode(
        cfg=cfg,
        pt_local=np_array(init_pt),
        run_tag=f"sweep_{idx}",
        save_artifacts=True,
        contact_candidates=[(name, np_array(values)) for name, values in contact_candidates],
    )
    summary = dict(episode.summary)
    summary["initial_contact_label"] = init_label
    summary["horizon_steps"] = int(horizon)
    summary["execute_chunk_steps"] = int(execute_chunk)
    summary["min_dubins_step_size"] = float(step_size)
    summary["online_refresh_every_steps"] = int(online_every)
    summary["status"] = "ok"
    summary["idx"] = int(idx)
    summary["score"] = score_summary(summary, weights_dict)
    write_json(str(case_dir / "summary_scored.json"), summary)
    return summary


def build_case_dir(payload: tuple) -> Path:
    idx, case, _base_cfg, _contact_candidates, _weights_dict, output_dir = payload
    _seed, horizon, execute_chunk, step_size, mu, rho, online_every = case
    return Path(output_dir) / (
        f"{idx:04d}_multictc_h{horizon}_c{execute_chunk}_"
        f"s{step_size:.1f}_mu{mu:.2f}_rho{rho:.1f}_u{online_every}"
    )


def failure_summary_from_payload(payload: tuple, err: BaseException) -> dict:
    idx, case, _base_cfg, _contact_candidates, _weights_dict, _output_dir = payload
    seed, horizon, execute_chunk, step_size, mu, rho, online_every = case
    return {
        "idx": int(idx),
        "status": "error",
        "seed": int(seed),
        "horizon_steps": int(horizon),
        "execute_chunk_steps": int(execute_chunk),
        "min_dubins_step_size": float(step_size),
        "mu": float(mu),
        "rho": float(rho),
        "online_refresh_every_steps": int(online_every),
        "error_type": type(err).__name__,
        "error_message": str(err),
        "coverage_final": 0.0,
        "coverage_max": 0.0,
        "clipping_fraction": 1.0,
        "max_contact_loss_streak": 9999,
        "final_model_pos_err": 9999.0,
        "replan_count_final": 9999,
        "score": -1.0e9,
    }


def ranked_successes(success_rows: list[dict]) -> list[dict]:
    ranked = sorted(success_rows, key=lambda item: float(item["score"]), reverse=True)
    return ranked[:10]

