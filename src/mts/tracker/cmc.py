import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from mts.common import CACHE, SPLITS
from mts.data import frame_paths, half_range, list_sequences, read_seqinfo

DOWNSCALE = 2
READ_FLAGS = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2, 4: cv2.IMREAD_REDUCED_GRAYSCALE_4}
FEATURES = {"maxCorners": 1000, "qualityLevel": 0.01, "minDistance": 1, "blockSize": 3}


def cmc_path(split: str, seq: str, cache: Path = CACHE) -> Path:
    return cache / "cmc" / split / f"{seq}.npy"


def read_grey(path: Path, downscale: int = DOWNSCALE) -> np.ndarray:
    img = cv2.imread(str(path), READ_FLAGS[downscale])
    if img is None:
        raise FileNotFoundError(path)
    return img


def estimate_affine(prev: np.ndarray, cur: np.ndarray, downscale: int = DOWNSCALE) -> np.ndarray:
    eye = np.eye(2, 3)
    pts = cv2.goodFeaturesToTrack(prev, mask=None, **FEATURES)
    if pts is None or len(pts) < 4:
        return eye
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None)
    ok = status.ravel() == 1
    if ok.sum() < 4:
        return eye
    affine, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC)
    if affine is None:
        return eye
    affine[:, 2] *= downscale
    return affine


def compute_sequence(seq: str, split: str, downscale: int = DOWNSCALE) -> tuple[np.ndarray, float, float]:
    cv2.setNumThreads(1)
    paths = frame_paths(seq, split)
    out = np.tile(np.eye(2, 3), (len(paths), 1, 1))
    decode_s = flow_s = 0.0
    prev = None
    for t, p in enumerate(paths):
        t0 = time.perf_counter()
        cur = read_grey(p, downscale)
        t1 = time.perf_counter()
        if prev is not None:
            out[t] = estimate_affine(prev, cur, downscale)
        flow_s += time.perf_counter() - t1
        decode_s += t1 - t0
        prev = cur
    return out, decode_s, flow_s


def _cache_one(args: tuple[str, str, int, Path]) -> dict:
    seq, split, downscale, cache = args
    affines, decode_s, flow_s = compute_sequence(seq, split, downscale)
    path = cmc_path(split, seq, cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, affines)
    meta = {"seq": seq, "split": split, "frames": len(affines), "downscale": downscale,
            "decode_ms_per_frame": 1000 * decode_s / len(affines), "cmc_ms_per_frame": 1000 * flow_s / len(affines)}
    path.with_suffix(".json").write_text(json.dumps(meta))
    return meta


def cache_all(splits: tuple[str, ...] = SPLITS, downscale: int = DOWNSCALE, cache: Path = CACHE,
              workers: int = 7, force: bool = False) -> list[dict]:
    jobs = [(seq, split, downscale, cache) for split in splits for seq in list_sequences()
            if force or not cmc_path(split, seq, cache).with_suffix(".json").exists()]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_cache_one, jobs))


def derive_subsplit(split: str, seq: str, cache: Path = CACHE) -> None:
    first, last = half_range(read_seqinfo(seq).length, split)
    parent = cmc_path("train_half", seq, cache)
    affines = np.load(parent)[first - 1:last].copy()
    affines[0] = np.eye(2, 3)
    path = cmc_path(split, seq, cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, affines)
    meta = json.loads(parent.with_suffix(".json").read_text()) | {"split": split, "frames": len(affines)}
    path.with_suffix(".json").write_text(json.dumps(meta))


def load_cmc(split: str, seq: str, cache: Path = CACHE) -> np.ndarray:
    return np.load(cmc_path(split, seq, cache))


def cmc_timing(split: str, seqs: list[str], cache: Path = CACHE) -> dict[str, float]:
    metas = [json.loads(cmc_path(split, s, cache).with_suffix(".json").read_text()) for s in seqs]
    frames = sum(m["frames"] for m in metas)
    return {k: sum(m[k] * m["frames"] for m in metas) / frames for k in ("decode_ms_per_frame", "cmc_ms_per_frame")}


def warp_state(mean: np.ndarray, cov: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot = affine[:, :2]
    scale = float(np.sqrt(abs(np.linalg.det(rot))))
    jac = np.zeros((8, 8))
    for k in (0, 4):
        jac[k:k + 2, k:k + 2] = rot
        jac[k + 2, k + 2] = 1.0
        jac[k + 3, k + 3] = scale
    new_mean = jac @ mean
    new_mean[:2] += affine[:, 2]
    return new_mean, jac @ cov @ jac.T


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache per-frame global affine camera motion (sparse optical flow)")
    parser.add_argument("--downscale", type=int, choices=sorted(READ_FLAGS), default=DOWNSCALE)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for m in cache_all(downscale=args.downscale, workers=args.workers, force=args.force):
        print(m)


if __name__ == "__main__":
    main()
