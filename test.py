import argparse
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from agcast.metrics import RadarMetricAccumulator, evaluate_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved radar predictions")
    parser.add_argument("--config", required=True, help="Dataset configuration JSON")
    parser.add_argument("--predictions", required=True, help="Prediction NPY array [B,T,1,H,W]")
    parser.add_argument("--targets", required=True, help="Observation NPY array in the same normalized space")
    parser.add_argument("--output", default="outputs/eval", help="Output directory for evaluation results")
    parser.add_argument("--batch-size", type=int, default=16, help="Samples processed per evaluation batch")
    parser.add_argument("--categorical-only", action="store_true", help="Compute CSI, HSS, BIAS and pooled CSI")
    return parser.parse_args()


def write_json(path, value):
    def convert(item):
        if isinstance(item, dict):
            return {str(key): convert(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(val) for val in item]
        if hasattr(item, "tolist"):
            return convert(item.tolist())
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    Path(path).write_text(json.dumps(convert(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_arrays(prediction_path, target_path):
    arrays = []
    for path in (prediction_path, target_path):
        path = Path(path).expanduser()
        if path.suffix.lower() != ".npy":
            raise ValueError(f"Expected a .npy array file, got: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
            raise ValueError(f"Expected a real numeric array: {path}")
        arrays.append(array)
    return arrays


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    name = config["dataname"]
    if name not in ("sevir", "meteonet", "shanghai", "cikm"):
        raise ValueError(f"Unsupported dataset: {name}")

    reference = RadarMetricAccumulator(name)
    for key, expected in (("physical_scale", reference.scale), ("physical_offset", reference.offset)):
        if key in config and not np.isclose(float(config[key]), expected, rtol=0, atol=1e-10):
            raise ValueError(f"{key} does not match the evaluation convention for {name}")

    pred, target = load_arrays(args.predictions, args.targets)
    expected = (int(config["aft_seq_length"]), *tuple(config["in_shape"])[1:])
    if pred.ndim != 5 or pred.shape != target.shape or tuple(pred.shape[1:]) != expected:
        raise ValueError(f"Expected matching [B,{','.join(map(str, expected))}] arrays; got {pred.shape} and {target.shape}")

    requested = ("radar_metrics",) if args.categorical_only else ("ssim", "radar_metrics")
    batches = math.ceil(len(pred) / args.batch_size)
    results, log = evaluate_predictions(
        pred, target, dataset_name=name, batch_size=args.batch_size, metrics=requested,
        progress=lambda iterable: tqdm(iterable, total=batches, desc=f"Evaluating {name}"),
    )

    write_json(output / "metrics.json", results)
    (output / "metrics.txt").write_text(log + "\n", encoding="utf-8")
    print(log)
    print(f"Evaluated {len(pred)} samples. Results saved to {output.resolve()}")


if __name__ == "__main__":
    main()
