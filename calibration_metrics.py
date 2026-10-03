"""Score binary probabilities against a prepared dataset (standard library only)."""

import argparse
import json
import math
from itertools import groupby
from pathlib import Path
import random


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def score(labels: list[int], probabilities: list[float], bins: int = 10):
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be nonempty and have equal length")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("labels must be binary")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in probabilities):
        raise ValueError("probabilities must be finite values in [0, 1]")

    n = len(labels)
    brier = sum((p - y) ** 2 for y, p in zip(labels, probabilities)) / n
    epsilon = 1e-15
    log_loss = -sum(
        y * math.log(max(epsilon, min(1 - epsilon, p)))
        + (1 - y) * math.log(max(epsilon, min(1 - epsilon, 1 - p)))
        for y, p in zip(labels, probabilities)
    ) / n
    accuracy = sum((p >= 0.5) == bool(y) for y, p in zip(labels, probabilities)) / n

    bucket = [[0, 0.0, 0.0] for _ in range(bins)]
    for y, p in zip(labels, probabilities):
        index = min(int(p * bins), bins - 1)
        bucket[index][0] += 1
        bucket[index][1] += p
        bucket[index][2] += y
    reliability = [
        {
            "bin": index,
            "count": count,
            "mean_probability": sum_p / count if count else None,
            "observed_rate": sum_y / count if count else None,
        }
        for index, (count, sum_p, sum_y) in enumerate(bucket)
    ]
    ece = sum(
        abs(sum_y - sum_p) for count, sum_p, sum_y in bucket if count
    ) / n

    # Evaluate at distinct score thresholds so equal scores have no arbitrary order.
    cumulative = [0.0]
    for probability, group in groupby(
        sorted(zip(probabilities, labels)), key=lambda pair: pair[0]
    ):
        cumulative.append(cumulative[-1] + sum(y - probability for _, y in group) / n)
    ecce_mad = max(abs(value) for value in cumulative)
    ecce_r = max(cumulative) - min(cumulative)

    return {
        "n": n,
        "positive_rate": sum(labels) / n,
        "accuracy_at_0_5": accuracy,
        "brier": brier,
        "log_loss": log_loss,
        "ece_10_equal_width_bins": ece,
        "ecce_mad": ecce_mad,
        "ecce_r": ecce_r,
        "reliability_bins": reliability,
    }


def ecce_r_null(probabilities: list[float], observed: float, draws: int, seed: int = 20261002):
    """Simulate perfectly calibrated Bernoulli outcomes conditional on these scores."""
    rng = random.Random(seed)
    ordered = sorted(probabilities)
    n = len(ordered)
    simulated = []
    for _ in range(draws):
        cumulative = low = high = 0.0
        simulated_labels = [int(rng.random() < probability) for probability in ordered]
        for probability, group in groupby(
            zip(ordered, simulated_labels), key=lambda pair: pair[0]
        ):
            cumulative += sum(y - probability for _, y in group) / n
            low = min(low, cumulative)
            high = max(high, cumulative)
        simulated.append(high - low)
    simulated.sort()
    return {
        "draws": draws,
        "null_95th_percentile": simulated[int(0.95 * (draws - 1))],
        "null_tail_fraction": (1 + sum(value >= observed for value in simulated)) / (draws + 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Prepared JSONL with id and label")
    parser.add_argument("predictions", type=Path, help="JSONL with id and probability")
    parser.add_argument("--null-draws", type=int, default=1000)
    args = parser.parse_args()
    dataset_rows = read_jsonl(args.dataset)
    prediction_rows = read_jsonl(args.predictions)
    by_id = {row["id"]: row for row in prediction_rows}
    if len(by_id) != len(prediction_rows):
        raise ValueError("Duplicate IDs in predictions")
    expected = {row["id"] for row in dataset_rows}
    if set(by_id) != expected:
        raise ValueError(
            f"Prediction IDs do not match dataset: {len(expected - set(by_id))} missing, "
            f"{len(set(by_id) - expected)} extra"
        )
    labels = [row["label"] for row in dataset_rows]
    probabilities = [float(by_id[row["id"]]["probability"]) for row in dataset_rows]
    metrics = score(labels, probabilities)
    if args.null_draws > 0:
        metrics["ecce_r_calibrated_null"] = ecce_r_null(
            probabilities, metrics["ecce_r"], args.null_draws
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
