from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_TASK_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def aggregate_results(results_root: Path, task_suites: tuple[str, ...]) -> dict:
    overall = {
        "total_count": 0,
        "success_count": 0,
        "success_rate": 0.0,
        "suite_success_rate_mean": 0.0,
    }
    suites = {}

    for suite in task_suites:
        results_file = results_root / suite / "evaluation_results.json"
        with open(results_file, "r", encoding="utf-8") as f:
            result = json.load(f)

        suite_result = {
            "total_count": int(result["total_episodes"]),
            "success_count": int(result["total_successes"]),
            "success_rate": float(result["total_success_rate"]),
        }
        suites[suite] = suite_result
        overall["total_count"] += suite_result["total_count"]
        overall["success_count"] += suite_result["success_count"]

    overall["success_rate"] = _safe_rate(overall["success_count"], overall["total_count"])
    overall["suite_success_rate_mean"] = _mean([item["success_rate"] for item in suites.values()])

    return {
        "overall": overall,
        "suites": suites,
    }


def _safe_rate(success_count: int, total_count: int) -> float:
    if total_count == 0:
        return 0.0
    return float(success_count) / float(total_count)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--suites", nargs="+", default=DEFAULT_TASK_SUITES)
    args = parser.parse_args()

    aggregated = aggregate_results(args.results_root, tuple(args.suites))
    output_path = args.results_root / "overall_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Aggregated results saved to {output_path}")
    print(json.dumps(aggregated["overall"], indent=2))


if __name__ == "__main__":
    main()
