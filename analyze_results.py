"""Analyze matched Jev/Clef JSONL predictions, including paired uncertainty.

Usage:
  python3 analyze_results.py DATASET.jsonl JEV.jsonl CLEF.jsonl > summary.json

The three files must contain the same row IDs. No API keys are needed here.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics

from calibration_metrics import ecce_r_null, read_jsonl, score


def matched_rows(dataset_path: Path, jev_path: Path, clef_path: Path):
    data = read_jsonl(dataset_path)
    if not data:
        raise ValueError("Dataset is empty")
    ids = [row["id"] for row in data]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs in dataset")
    predictions = {}
    for name, path in (("jev", jev_path), ("clef", clef_path)):
        rows = read_jsonl(path)
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != set(ids):
            raise ValueError(f"{name} predictions do not match dataset IDs")
        predictions[name] = by_id
    return data, predictions


def threshold_counts(labels, probabilities):
    tp = sum(y == 1 and p >= 0.5 for y, p in zip(labels, probabilities))
    fp = sum(y == 0 and p >= 0.5 for y, p in zip(labels, probabilities))
    fn = sum(y == 1 and p < 0.5 for y, p in zip(labels, probabilities))
    tn = sum(y == 0 and p < 0.5 for y, p in zip(labels, probabilities))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def paired_bootstrap(labels, jev, clef, draws, seed):
    if draws < 1:
        return None
    rng = random.Random(seed)
    differences = {"brier": [], "ecce_r": []}
    n = len(labels)
    for _ in range(draws):
        indices = [rng.randrange(n) for _ in range(n)]
        sampled_labels = [labels[i] for i in indices]
        j = score(sampled_labels, [jev[i] for i in indices])
        c = score(sampled_labels, [clef[i] for i in indices])
        for key in differences:
            differences[key].append(j[key] - c[key])
    original_j, original_c = score(labels, jev), score(labels, clef)
    output = {}
    for key, values in differences.items():
        values.sort()
        # Nearest-rank central 95% interval; 2,000 draws use indices 49, 1949.
        low = max(0, int(0.025 * draws) - 1)
        high = max(0, int(0.975 * draws) - 1)
        output[key] = {"difference": original_j[key] - original_c[key],
                       "ci_95": [values[low], values[high]]}
    return output


def analyze(dataset_path, jev_path, clef_path, bootstrap_draws, null_draws, seed):
    data, predictions = matched_rows(dataset_path, jev_path, clef_path)
    ids = [row["id"] for row in data]
    labels = [int(row["label"]) for row in data]
    probabilities = {
        name: [float(rows[row_id]["probability"]) for row_id in ids]
        for name, rows in predictions.items()
    }
    output = {"dataset": data[0]["dataset"], "n": len(data),
              "positive_rate": sum(labels) / len(labels), "models": {}}
    for name in ("jev", "clef"):
        rows = predictions[name]
        p = probabilities[name]
        metrics = score(labels, p)
        model = {
            "model_ids": dict(Counter(row["model"] for row in rows.values())),
            "mean_probability": sum(p) / len(p),
            "median_latency_ms": statistics.median(row["elapsed_ms"] for row in rows.values()),
            "at_0_5": threshold_counts(labels, p),
            "metrics": metrics,
        }
        if null_draws > 0:
            model["ecce_r_calibrated_null"] = ecce_r_null(
                p, metrics["ecce_r"], null_draws, seed
            )
        output["models"][name] = model
    output["paired_bootstrap_jev_minus_clef"] = paired_bootstrap(
        labels, probabilities["jev"], probabilities["clef"], bootstrap_draws, seed
    )
    if all("source" in row for row in data):
        groups = defaultdict(list)
        for i, row in enumerate(data):
            groups[row["source"]].append(i)
        output["sources"] = {}
        for source, indices in groups.items():
            y = [labels[i] for i in indices]
            output["sources"][source] = {"n": len(indices)}
            for name in ("jev", "clef"):
                p = [probabilities[name][i] for i in indices]
                m = score(y, p)
                output["sources"][source][name] = {
                    key: m[key] for key in ("accuracy_at_0_5", "brier", "ecce_r")
                }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("jev", type=Path)
    parser.add_argument("clef", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--null-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    print(json.dumps(analyze(args.dataset, args.jev, args.clef,
                             args.bootstrap_draws, args.null_draws, args.seed),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
