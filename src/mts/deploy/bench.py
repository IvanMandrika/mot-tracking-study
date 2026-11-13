import argparse
import json
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mts import plotting
from mts.common import RESULTS
from mts.data import frame_paths, list_sequences, read_seqinfo, save_dets, split_length
from mts.deploy.export import CPU_THREADS, OUTPUTS, VARIANTS, cast_nodes, make_session, model_path, split_sizes
from mts.detector import FT_CHECKPOINT, LONG_SIDE, Detector, coco_weights, input_size, person_detections, \
    person_label, read_rgb, to_tensor

TORCH = "torch_cuda_fp32"
ORDER = [TORCH, *VARIANTS]
LABELS = {TORCH: "PyTorch CUDA fp32", "ort_cuda_fp32": "ORT CUDA fp32", "ort_cuda_fp16": "ORT CUDA fp16",
          "ort_cuda_fp16_exhaustive": "ORT CUDA fp16 (exhaustive)", "ort_cpu_fp32": "ORT CPU fp32",
          "ort_cpu_int8": "ORT CPU INT8"}
COLOURS = dict(zip(ORDER, [plotting.SERIES[i] for i in (2, 3, 4, 5, 6, 7)]))
SPLIT = "val_half"
TRACK_METRICS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]
MIN_Y_SPAN = 2.0
TIMING_ONLY = {"ort_cuda_fp16_exhaustive"}
Runner = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def time_calls(fn: Callable, inputs: Sequence, warmup: int = 20) -> dict[str, float]:
    for i in range(warmup):
        fn(inputs[i % len(inputs)])
    times = np.empty(len(inputs))
    for i, x in enumerate(inputs):
        t0 = time.perf_counter()
        fn(x)
        times[i] = time.perf_counter() - t0
    ms = 1000 * times
    return {"latency_median_ms": float(np.median(ms)), "latency_p90_ms": float(np.percentile(ms, 90)),
            "latency_mean_ms": float(ms.mean()), "latency_std_ms": float(ms.std()), "latency_frames": len(ms)}


def gpu_processes() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip() and int(line.split(",")[0]) != os.getpid()]


class Runners:
    def __init__(self, variant: str, weights: Path | str, tag: str, threads: int) -> None:
        self.variant = variant
        self.tag = tag
        self.threads = threads
        self.sessions: dict[tuple[int, int], object] = {}
        self.det = Detector.load(weights) if variant == TORCH else None

    def get(self, size: tuple[int, int]) -> Runner:
        if self.det is not None:
            return lambda x: self.det.forward(torch.from_numpy(x))
        if size not in self.sessions:
            precision, device, conv_algo = VARIANTS[self.variant]
            self.sessions[size] = make_session(model_path(precision, size, self.tag), device, self.threads, conv_algo)
        sess = self.sessions[size]
        return lambda x: tuple(sess.run(list(OUTPUTS), {"pixel_values": x}))

    def close(self) -> None:
        self.sessions.clear()
        if self.det is not None:
            self.det.close()
            self.det = None


def latency_inputs(n: int, seed: int, long_side: int = LONG_SIDE) -> tuple[tuple[int, int], list[np.ndarray]]:
    size, seqs = max(split_sizes(long_side).items(), key=lambda kv: len(kv[1]))
    paths = [p for s in seqs for p in frame_paths(s, SPLIT)]
    pick = np.random.default_rng(seed).choice(len(paths), size=n, replace=False)
    return size, [to_tensor(read_rgb(paths[i]), size)[None].numpy() for i in pick]


def detect_split(runners: Runners, person: int, source: str, long_side: int = LONG_SIDE) -> int:
    total = 0
    for seq in list_sequences():
        info = read_seqinfo(seq)
        size = input_size(info.width, info.height, long_side)
        run = runners.get(size)
        rows = []
        for f, path in enumerate(frame_paths(seq, SPLIT), start=1):
            logits, boxes = run(to_tensor(read_rgb(path), size)[None].numpy())
            d = person_detections(logits[0], boxes[0], person, info.width, info.height)
            rows.append(np.concatenate([np.full((len(d), 1), f, np.float32), d], axis=1))
        dets = np.concatenate(rows)
        save_dets(dets, source, SPLIT, seq)
        total += len(dets)
        print(f"{source} {seq}: {split_length(info.length, SPLIT)} frames, {len(dets)} dets", flush=True)
    return total


def tracker_config(source: str):
    from mts.tracker.tracker import TrackerConfig

    per_source = json.loads((RESULTS / "tracking_best.json").read_text())["per_source"]
    if source not in per_source:
        raise SystemExit(f"no selected tracker for {source} in tracking_best.json, run mts.exp_tracking first")
    return per_source[source]["variant"], TrackerConfig(**per_source[source]["config"])


def track_quality(source: str, cfg, name: str) -> dict[str, float]:
    from mts.evaluate import COMBINED
    from mts.exp_tracking import TRACKS, reid_ready, run_config

    if cfg.reid and not reid_ready(source, SPLIT):
        from mts.tracker.reid import Embedder, cache_source

        emb = Embedder("cuda")
        cache_source(emb, source, (SPLIT,))
        del emb
        torch.cuda.empty_cache()
    res, _, _ = run_config(cfg, source, SPLIT, TRACKS / source / SPLIT / f"deploy_{name}")
    return {k: res[COMBINED][k] for k in TRACK_METRICS}


def model_mb(variant: str, tag: str, size: tuple[int, int]) -> float:
    if variant == TORCH:
        return (FT_CHECKPOINT / "model.safetensors").stat().st_size / 2**20 if tag == "rtdetr_ft" else float("nan")
    return model_path(VARIANTS[variant][0], size, tag).stat().st_size / 2**20


def time_variant(variant: str, weights: Path | str, tag: str, latency: tuple[tuple[int, int], list], threads: int,
                 warmup: int) -> dict[str, float]:
    size, inputs = latency
    cpu = variant != TORCH and VARIANTS[variant][1] == "cpu"
    all_cores = os.sched_getaffinity(0)
    if cpu:
        os.sched_setaffinity(0, set(sorted(all_cores)[:threads]))
    load_before = os.getloadavg()[0]
    runners = Runners(variant, weights, tag, threads)
    try:
        stats = time_calls(runners.get(size), inputs, warmup)
    finally:
        runners.close()
        os.sched_setaffinity(0, all_cores)
    extra = {"model_mb": model_mb(variant, tag, size), "loadavg_1min_before": load_before,
             "loadavg_1min_after": os.getloadavg()[0], "pinned_cores": float(threads if cpu else 0)}
    if variant != TORCH:
        extra["onnx_cast_nodes"] = float(cast_nodes(model_path(VARIANTS[variant][0], size, tag)))
    return {**stats, **extra}


def detect_variant(variant: str, weights: Path | str, tag: str, person: int, threads: int) -> None:
    runners = Runners(variant, weights, tag, threads)
    try:
        detect_split(runners, person, variant)
    finally:
        runners.close()


def all_cached(source: str) -> bool:
    from mts.data import dets_path

    return all(dets_path(source, SPLIT, s).exists() for s in list_sequences())


def quality(variant: str, tag: str, cfg, cfg_name: str) -> dict[str, float]:
    from mts.detect import evaluate_source

    source = tag if variant == TORCH else variant
    return {**evaluate_source(source, SPLIT), **track_quality(source, cfg, cfg_name)}


def to_long(results: dict[str, dict[str, float]], threads: int, size: tuple[int, int], cfg_name: str,
            tag: str) -> pd.DataFrame:
    rows = []
    for variant, metrics in results.items():
        precision, device, conv_algo = ("fp32", "cuda", "-") if variant == TORCH else VARIANTS[variant]
        for k, v in metrics.items():
            rows.append({"variant": variant, "runtime": "pytorch" if variant == TORCH else "onnxruntime",
                         "device": device, "precision": precision, "conv_algo": conv_algo,
                         "cpu_threads": threads if device == "cpu" else 0,
                         "weights": tag, "latency_input": f"{size[0]}x{size[1]}", "split": SPLIT,
                         "tracker": cfg_name, "metric": k, "value": v})
    return pd.DataFrame(rows)


def place_labels(ax, points: list[tuple[float, float, str, str]]) -> None:
    from matplotlib.transforms import Bbox

    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    placed = []
    for x, y, _, _ in points:
        px, py = ax.transData.transform((x, y))
        placed.append(Bbox([[px - 7, py - 7], [px + 7, py + 7]]))
    offsets = [(dx, dy) for dy in (4, -14, 16, -26, 28, -38) for dx in (7, -7)]
    for x, y, text, color in sorted(points):
        for dx, dy in offsets:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=color,
                            ha="left" if dx > 0 else "right")
            box = t.get_window_extent(renderer).expanded(1.05, 1.15)
            inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
            if inside and not any(box.overlaps(b) for b in placed):
                placed.append(box)
                break
            t.remove()


def plot_tradeoff(df: pd.DataFrame) -> None:
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    plotting.setup()
    wide = df.pivot_table(index="variant", columns="metric", values="value")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, metric in zip(axes, ["HOTA", "AP50"]):
        points = []
        for variant in ORDER:
            if variant not in wide.index or np.isnan(wide.loc[variant].get(metric, np.nan)):
                continue
            x, y = wide.loc[variant, "latency_median_ms"], wide.loc[variant, metric]
            color = COLOURS[variant]
            ax.scatter([x], [y], s=55, color=color, zorder=3)
            points.append((x, y, LABELS[variant], color))
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 40, 70, 100, 150, 200, 300, 500, 1000, 2000]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("median latency per frame, batch 1 (ms, log scale)")
        ax.set_ylabel(f"{metric} (VAL-HALF)")
        xs = [pt[0] for pt in points]
        ax.set_xlim(min(xs) / 1.3, max(xs) * 2.4)
        lo, hi = min(pt[1] for pt in points), max(pt[1] for pt in points)
        pad = max(0.0, (MIN_Y_SPAN - (hi - lo)) / 2)
        ax.set_ylim(lo - pad - 0.1 * MIN_Y_SPAN, hi + pad + 0.1 * MIN_Y_SPAN)
        fig.canvas.draw()
        place_labels(ax, points)
    axes[0].set_title("Tracking HOTA vs detector latency")
    axes[1].set_title("Detection AP50 vs detector latency")
    fig.tight_layout()
    plotting.save(fig, "deploy_tradeoff.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency and VAL-HALF quality of PyTorch and ONNX Runtime variants")
    parser.add_argument("--variants", nargs="+", choices=ORDER, default=ORDER)
    parser.add_argument("--weights", type=Path, default=None, help="default: fine-tuned checkpoint if present")
    parser.add_argument("--tag", default=None, help="detection source name of the weights, default rtdetr_ft or rtdetr_coco")
    parser.add_argument("--latency-frames", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--threads", type=int, default=CPU_THREADS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--redetect", action="store_true")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--out", type=Path, default=RESULTS / "deploy.csv")
    args = parser.parse_args()
    busy = gpu_processes()
    print("GPU compute processes before timing:", busy or "none", flush=True)
    if busy and not args.allow_busy_gpu:
        raise SystemExit("another process uses the GPU, timings would be invalid")
    ft = (FT_CHECKPOINT / "model.safetensors").exists()
    weights = args.weights or (FT_CHECKPOINT if ft else coco_weights())
    tag = args.tag or ("rtdetr_ft" if Path(weights).resolve() == FT_CHECKPOINT.resolve() else "rtdetr_coco")
    torch.set_num_threads(args.threads)
    from transformers import AutoConfig

    person = person_label(AutoConfig.from_pretrained(str(weights)).id2label)
    latency = latency_inputs(args.latency_frames, args.seed)
    results = {}
    for variant in args.variants:
        results[variant] = time_variant(variant, weights, tag, latency, args.threads, args.warmup)
        print(variant, {k: round(v, 2) for k, v in results[variant].items()}, flush=True)
    busy = gpu_processes()
    print("GPU compute processes after timing:", busy or "none", flush=True)
    for variant in args.variants:
        results[variant]["other_gpu_processes_after_timing"] = len(busy)
        if variant != TORCH and variant not in TIMING_ONLY and (args.redetect or not all_cached(variant)):
            detect_variant(variant, weights, tag, person, args.threads)
    cfg_name, cfg = tracker_config(tag)
    print("tracker:", cfg_name, cfg, flush=True)
    for variant in (v for v in args.variants if v not in TIMING_ONLY):
        results[variant].update(quality(variant, tag, cfg, cfg_name))
        print(variant, {k: round(v, 2) for k, v in results[variant].items()}, flush=True)
    df = to_long(results, args.threads, latency[0], cfg_name, tag)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    plot_tradeoff(df)
    keys = ["latency_median_ms", "latency_p90_ms", "AP", "AP50", "HOTA", "IDF1", "MOTA", "IDSW", "model_mb"]
    table = df.pivot_table(index="variant", columns="metric", values="value").reindex(args.variants)[keys]
    pd.set_option("display.width", 200)
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
