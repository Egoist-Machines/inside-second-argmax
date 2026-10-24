"""Reproduce the fixed successful record-switch trace with the native weights."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "67931e9f632b1f8c489091323ae4baeba8c5d01d41cd054e9d75440f4a3798fa"
COMPARISONS = (1, 6, 7, 8)
PREFIX = [1, 0, 0, 0, 0, 0, 0, 0.5, 0]


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def step(model, old, x):
    pre = [dot(row, old) + u * x
           for row, u in zip(model["recurrent"], model["input"])]
    return [max(0.0, value) for value in pre]


def describe(model, hidden, label, inputs, native):
    readout = model["readout"]
    mean = [sum(row[j] for row in readout) / 10 for j in range(16)]
    centered = [[value - mean[j] for j, value in enumerate(row)]
                for row in readout]
    scores = [dot(row, hidden) for row in readout]
    comparisons = [sum(row[i] * hidden[i] for i in COMPARISONS)
                   for row in centered]
    remainder = [sum(row[i] * hidden[i] for i in range(16)
                     if i not in COMPARISONS) for row in centered]
    target = sorted(range(10), key=lambda i: inputs[i], reverse=True)[1]
    return {
        "label": label,
        "inputs": inputs,
        "native": native,
        "target": target,
        "prediction": max(range(10), key=lambda i: scores[i]),
        "hidden": hidden,
        "scores": scores,
        "centered_comparison_scores": comparisons,
        "centered_other_scores": remainder,
        "margin_0_minus_7": scores[0] - scores[7],
        "comparison_margin_0_minus_7": comparisons[0] - comparisons[7],
        "other_margin_0_minus_7": remainder[0] - remainder[7],
        "comparison_activity": [hidden[i] for i in COMPARISONS],
        "record_activity": hidden[4],
    }


def calculate():
    source = (ROOT / "results/model.json").read_bytes()
    assert hashlib.sha256(source).hexdigest() == MODEL_SHA256
    model = json.loads(source)
    old = [0.0] * 16
    for x in PREFIX:
        old = step(model, old, x)
    low = step(model, old, 0.25)
    record = step(model, old, 2.0)
    restored = record[:]
    for i in COMPARISONS:
        restored[i] = low[i]
    cases = [
        describe(model, low, "Lower input", PREFIX + [0.25], True),
        describe(model, record, "New maximum", PREFIX + [2.0], True),
        describe(model, restored, "Restore lower-input comparisons",
                 PREFIX + [2.0], False),
    ]
    assert [case["prediction"] for case in cases] == [7, 0, 7]
    assert [case["target"] for case in cases] == [7, 0, 0]
    assert all(value > 0 for value in cases[0]["comparison_activity"])
    assert all(value == 0 for value in cases[1]["comparison_activity"])
    assert cases[0]["comparison_margin_0_minus_7"] < -18
    assert cases[1]["comparison_margin_0_minus_7"] == 0
    assert cases[0]["record_activity"] == 0 < cases[1]["record_activity"]
    assert cases[0]["margin_0_minus_7"] < -24
    assert cases[1]["margin_0_minus_7"] > 1
    assert cases[2]["margin_0_minus_7"] < -16
    assert cases[1]["other_margin_0_minus_7"] == cases[2]["other_margin_0_minus_7"]
    for case in cases:
        assert abs(case["margin_0_minus_7"]
                   - case["comparison_margin_0_minus_7"]
                   - case["other_margin_0_minus_7"]) < 1e-10
        mean_score = sum(case["scores"]) / 10
        for score, comparison, other in zip(
                case["scores"], case["centered_comparison_scores"],
                case["centered_other_scores"]):
            assert abs(score - mean_score - comparison - other) < 1e-10
    return {
        "schema_version": 1,
        "model_file_sha256": MODEL_SHA256,
        "checkpoint_sha256": model["checkpoint_sha256"],
        "prefix": PREFIX,
        "comparison_neurons": list(COMPARISONS),
        "scope": "Two native final-input branches and one internal readout control. The control replaces only final comparison activities; reachability of its mixed state is unproved. This fixed float64 trace is not an exact proof or a population claim.",
        "cases": cases,
    }


def compare(actual, saved, path="trace"):
    if isinstance(actual, dict):
        assert actual.keys() == saved.keys(), path
        for key in actual:
            compare(actual[key], saved[key], f"{path}.{key}")
    elif isinstance(actual, list):
        assert len(actual) == len(saved), path
        for i, (a, b) in enumerate(zip(actual, saved)):
            compare(a, b, f"{path}[{i}]")
    elif isinstance(actual, float):
        assert abs(actual - saved) < 1e-10, path
    else:
        assert actual == saved, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path,
                        default=ROOT / "results/successful-trace.json")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = calculate()
    if args.write:
        args.check.write_text(json.dumps(result, indent=2) + "\n")
    else:
        compare(result, json.loads(args.check.read_text()))
    print("Successful trace: native choices 7 and 0; restored comparisons choose 7.")


if __name__ == "__main__":
    main()
