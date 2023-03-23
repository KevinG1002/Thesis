import os
import os
import sys
import json


def get_max_or_min_per_metric(metrics_json: dict):
    best_metrics = {}
    for key, item in metrics_json.items():
        if "loss" in key:
            best_metrics[f"best_{key}"] = min(item)
        elif key == "mean_distinct_count" or key == "ensemble_confusion_matrix":
            pass
        else:
            best_metrics[f"best_{key}"] = max(item)
    pretty_json = json.dumps(best_metrics, indent=4)
    return pretty_json


def test(json_path):
    with open(json_path, "r") as f:
        results = json.load(f)
    print(get_max_or_min_per_metric(results))
    return


if __name__ == "__main__":
    json_file_path = sys.argv[1]
    test(json_file_path)
