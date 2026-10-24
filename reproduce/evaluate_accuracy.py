"""Sample the pinned native recurrence on independent standard-normal inputs."""

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def evaluate(samples=100_000, seed=0, batch_size=4096):
    if samples <= 0 or batch_size <= 0 or seed < 0:
        raise ValueError("Expected positive sample and batch counts and a nonnegative seed")
    source = (ROOT / "results/model.json").read_bytes()
    model = json.loads(source)
    u = np.asarray(model["input"], dtype=np.float64)
    W = np.asarray(model["recurrent"], dtype=np.float64)
    V = np.asarray(model["readout"], dtype=np.float64)
    assert (u.shape, W.shape, V.shape) == ((16,), (16, 16), (10, 16))
    rng = np.random.Generator(np.random.PCG64(seed))
    inputs = rng.standard_normal((samples, 10), dtype=np.float64)
    ordered = np.argsort(inputs, axis=1, kind="stable")
    sorted_inputs = np.take_along_axis(inputs, ordered, axis=1)
    assert not np.any(np.diff(sorted_inputs, axis=1) == 0), "Input ties require a declared target convention"
    targets = ordered[:, -2]
    correct = 0
    for start in range(0, samples, batch_size):
        x = inputs[start:start + batch_size]
        hidden = np.zeros((len(x), 16), dtype=np.float64)
        for t in range(10):
            hidden = np.maximum(hidden @ W.T + x[:, t, None] * u, 0)
        prediction = np.argmax(hidden @ V.T, axis=1)
        correct += int(np.count_nonzero(prediction == targets[start:start + len(x)]))
    p = correct / samples
    z = 1.959963984540054
    denominator = 1 + z * z / samples
    center = (p + z * z / (2 * samples)) / denominator
    half = z * math.sqrt(p * (1 - p) / samples + z * z / (4 * samples * samples)) / denominator
    return {
        "schema_version": 1,
        "method": "Empirical evaluation of the native recurrence; no mechanism-based accuracy estimate",
        "checkpoint_sha256": model["checkpoint_sha256"],
        "model_json_sha256": hashlib.sha256(source).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "settings": {"samples": samples, "seed": seed, "batch_size": batch_size,
                     "rng": "numpy.random.Generator(PCG64)", "distribution": "10 independent N(0,1) inputs",
                     "device": "CPU", "dtype": "float64", "prediction": "first maximum score",
                     "target": "position of the second-largest input; no sampled input ties"},
        "inputs_sha256": hashlib.sha256(inputs.astype("<f8", copy=False).tobytes()).hexdigest(),
        "correct": correct,
        "errors": samples - correct,
        "accuracy": p,
        "accuracy_wilson_95": [center - half, center + half],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path, help="Recompute and compare with a saved evaluation")
    args = parser.parse_args()
    saved = json.loads(args.check.read_text()) if args.check else None
    settings = saved["settings"] if saved else {"samples": 100_000, "seed": 0, "batch_size": 4096}
    actual = evaluate(*(getattr(args, key) if getattr(args, key) is not None else settings[key]
                        for key in ("samples", "seed", "batch_size")))
    if saved:
        checked_keys = set(actual) - {"python_version", "numpy_version"}
        for key in checked_keys:
            if actual[key] != saved[key]:
                raise AssertionError(f"Saved accuracy record differs at {key}")
        print(f"Accuracy record verified: {actual['correct']}/{actual['settings']['samples']} correct.")
    elif args.output:
        args.output.write_text(json.dumps(actual, indent=2, allow_nan=False) + "\n")
        print(f"{actual['correct']}/{actual['settings']['samples']} correct ({100*actual['accuracy']:.3f}%).")
    else:
        print(json.dumps(actual, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
