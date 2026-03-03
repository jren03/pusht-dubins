from __future__ import annotations

import csv
import json
from pathlib import Path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: dict) -> None:
    p = Path(path)
    ensure_parent(p)
    with p.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def write_jsonl(path: str, rows: list[dict]) -> None:
    p = Path(path)
    ensure_parent(p)
    with p.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, sort_keys=True))
            file_obj.write("\n")


def write_csv(path: str, rows: list[dict]) -> None:
    p = Path(path)
    ensure_parent(p)
    if len(rows) == 0:
        p.write_text("", encoding="utf-8")
        return
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = sorted(keys)
    normalized = [{key: row[key] if key in row else "" for key in fieldnames} for row in rows]
    with p.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)


def score_summary(summary: dict, weights: dict) -> float:
    return (
        weights["coverage_weight"] * float(summary["coverage_final"])
        - weights["clipping_penalty"] * float(summary["clipping_fraction"])
        - weights["contact_loss_penalty"] * float(summary["max_contact_loss_streak"]) / 20.0
        - weights["model_mismatch_penalty"] * float(summary["final_model_pos_err"]) / 100.0
        - weights["replan_penalty"] * float(summary["replan_count_final"]) / 50.0
    )

