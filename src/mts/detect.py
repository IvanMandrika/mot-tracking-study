import argparse
import contextlib
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from mts.common import CACHE, MOT17, RESULTS, SPLITS, SUBSPLITS
from mts.data import MIN_VISIBILITY, PEDESTRIAN, derive_subsplit_dets, dets_by_frame, dets_path, frame_paths, \
    list_sequences, load_dets, load_gt, pedestrian_mask, read_seqinfo, save_dets, split_length
from mts.detector import FT_CHECKPOINT, LONG_SIDE, TOP_K, Detector, coco_weights, input_size, read_rgb, to_tensor, \
    worker_init

PERSON_ON_VEHICLE, STATIC_PERSON, DISTRACTOR, REFLECTION = 2, 7, 8, 12
DISTRACTOR_CLASSES = (PERSON_ON_VEHICLE, STATIC_PERSON, DISTRACTOR, REFLECTION)
SOURCES = ("public_frcnn", "rtdetr_coco", "rtdetr_ft")
GT_PROTOCOLS = ("vis025", "all_vis")
EVAL_SPLITS = ("val_half", "train_holdout", "train_fit")


class FrameDataset(Dataset):
    def __init__(self, paths: list[Path], size: tuple[int, int]) -> None:
        self.paths = paths
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        return to_tensor(read_rgb(self.paths[i]), self.size)


def frame_loader(paths: list[Path], size: tuple[int, int], batch: int, workers: int) -> DataLoader:
    return DataLoader(FrameDataset(paths, size), batch_size=batch, num_workers=workers, pin_memory=True,
                      worker_init_fn=worker_init)


def detect_frames(det: Detector, seq: str, frames: list[int], split: str, batch: int = 4,
                  workers: int = 6) -> np.ndarray:
    info = read_seqinfo(seq)
    paths = frame_paths(seq, split)
    size = input_size(info.width, info.height, det.long_side)
    rows = []
    loader = frame_loader([paths[f - 1] for f in frames], size, batch, workers)
    it = iter(frames)
    for x in loader:
        for d in det.detect_batch(x, info.width, info.height):
            rows.append(np.concatenate([np.full((len(d), 1), next(it), np.float32), d], axis=1))
    return np.concatenate(rows) if rows else np.zeros((0, 6), np.float32)


def cache_source(det: Detector, source: str, splits: tuple[str, ...], batch: int, workers: int) -> pd.DataFrame:
    rows = []
    for split in splits:
        for seq in list_sequences():
            n = split_length(read_seqinfo(seq).length, split)
            t0 = time.perf_counter()
            dets = detect_frames(det, seq, list(range(1, n + 1)), split, batch, workers)
            dt = time.perf_counter() - t0
            save_dets(dets, source, split, seq)
            rows.append({"source": source, "split": split, "seq": seq, "frames": n, "dets": len(dets),
                         "fps": n / dt})
            print(f"{source} {split} {seq}: {n} frames, {len(dets)} dets, {n / dt:.1f} fps", flush=True)
    return pd.DataFrame(rows)


def eval_targets(seq: str, split: str, protocol: str = "vis025",
                 root: Path = MOT17) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = load_gt(seq, split, root)
    if protocol == "vis025":
        pos = pedestrian_mask(gt, MIN_VISIBILITY)
        return gt[pos], gt[~pos & gt["cls"].isin((PEDESTRIAN, *DISTRACTOR_CLASSES))]
    if protocol == "all_vis":
        return gt[pedestrian_mask(gt)], gt[gt["cls"].isin(DISTRACTOR_CLASSES)]
    raise ValueError(f"unknown GT protocol {protocol!r}, expected one of {GT_PROTOCOLS}")


def boxes_per_frame(df: pd.DataFrame, frames: list[int]) -> list[np.ndarray]:
    groups = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in df.groupby("frame")}
    return [groups.get(f, np.zeros((0, 4))) for f in frames]


def coco_metrics(gts: list[np.ndarray], ignores: list[np.ndarray], dets: list[np.ndarray],
                 max_dets: int = TOP_K) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    anns = []
    for i, (g, ig) in enumerate(zip(gts, ignores)):
        for boxes, crowd in ((g, 0), (ig, 1)):
            for x1, y1, x2, y2 in boxes:
                w, h = float(x2 - x1), float(y2 - y1)
                anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 1, "bbox": [float(x1), float(y1), w, h],
                             "area": w * h, "iscrowd": crowd})
    res = [np.column_stack([np.full(len(d), i), d[:, 0], d[:, 1], d[:, 2] - d[:, 0], d[:, 3] - d[:, 1], d[:, 4],
                            np.ones(len(d))]) for i, d in enumerate(dets) if len(d)]
    if not res:
        return {"AP": 0.0, "AP50": 0.0, "recall50": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = {"images": [{"id": i} for i in range(len(gts))], "annotations": anns,
                        "categories": [{"id": 1, "name": "person"}]}
        coco.createIndex()
        ev = COCOeval(coco, coco.loadRes(np.concatenate(res)), "bbox")
        ev.params.maxDets = [max_dets]
        ev.params.areaRng = [[0.0, 1e10]]
        ev.params.areaRngLbl = ["all"]
        ev.evaluate()
        ev.accumulate()
    prec = ev.eval["precision"][:, :, 0, 0, 0]
    return {"AP": 100 * float(prec[prec > -1].mean()), "AP50": 100 * float(prec[0][prec[0] > -1].mean()),
            "recall50": 100 * float(ev.eval["recall"][0, 0, 0, 0])}


def split_targets(split: str, frames: dict[str, list[int]] | None = None,
                  protocol: str = "vis025") -> tuple[list, list]:
    gts, igns = [], []
    for seq in list_sequences():
        fr = frames[seq] if frames else list(range(1, split_length(read_seqinfo(seq).length, split) + 1))
        pos, ign = eval_targets(seq, split, protocol)
        gts += boxes_per_frame(pos, fr)
        igns += boxes_per_frame(ign, fr)
    return gts, igns


def evaluate_source(source: str, split: str, protocol: str = "vis025") -> dict[str, float]:
    gts, igns = split_targets(split, protocol=protocol)
    dets = []
    for seq in list_sequences():
        n = split_length(read_seqinfo(seq).length, split)
        dets += dets_by_frame(load_dets(source, split, seq), n)
    return coco_metrics(gts, igns, dets)


def available_sources(split: str) -> list[str]:
    root = CACHE / "dets"
    names = sorted(p.name for p in root.iterdir()) if root.exists() else []
    ordered = [s for s in SOURCES if s in names] + [s for s in names if s not in SOURCES]
    return [s for s in ordered if all(dets_path(s, split, q).exists() for q in list_sequences())]


def ensure_subsplit_dets(source: str) -> None:
    for split in SUBSPLITS:
        for seq in list_sequences():
            if dets_path(source, "train_half", seq).exists():
                derive_subsplit_dets(source, split, seq)


def visibility_stats(splits: list[str]) -> pd.DataFrame:
    rows = []
    for split in splits:
        gt = pd.concat([load_gt(seq, split) for seq in list_sequences()])
        ped = gt[pedestrian_mask(gt)]
        low = int((ped["vis"] < MIN_VISIBILITY).sum())
        rows += [{"split": split, "metric": "gt_pedestrian_boxes", "value": len(ped)},
                 {"split": split, "metric": "gt_boxes_vis_below_0.25", "value": low},
                 {"split": split, "metric": "fraction_vis_below_0.25", "value": round(low / len(ped), 4)}]
    return pd.DataFrame(rows)


def run_eval(splits: list[str], sources: list[str] | None, out: Path) -> pd.DataFrame:
    rows = []
    for source in sources or available_sources("val_half"):
        ensure_subsplit_dets(source)
        for split in splits:
            for protocol in GT_PROTOCOLS:
                m = evaluate_source(source, split, protocol)
                print(f"{source} {split} {protocol}: " + ", ".join(f"{k} {v:.2f}" for k, v in m.items()), flush=True)
                rows += [{"source": source, "split": split, "gt": protocol, "metric": k, "value": round(v, 3)}
                         for k, v in m.items()]
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    visibility_stats(splits).to_csv(out.with_name("gt_visibility.csv"), index=False)
    return df


def measure_resolution(long_sides: list[int], frames: dict[str, list[int]], batch: int, workers: int) -> pd.DataFrame:
    gts, igns = split_targets("train_half", frames)
    rows = []
    for ls in long_sides:
        det = Detector.load(coco_weights(), long_side=ls)
        torch.cuda.reset_peak_memory_stats()
        x = torch.rand(batch, 3, *input_size(1920, 1080, ls)[::-1])
        for _ in range(3):
            det.detect_batch(x, 1920, 1080)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 10
        for _ in range(reps):
            det.detect_batch(x, 1920, 1080)
        torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0) / (reps * batch)
        mem = torch.cuda.max_memory_allocated() / 2**20
        dets = []
        for seq, fr in frames.items():
            d = detect_frames(det, seq, fr, "train_half", batch, workers)
            d[:, 0] -= fr[0] - 1
            dets += dets_by_frame(d, len(fr))
        m = coco_metrics(gts, igns, dets)
        w, h = input_size(1920, 1080, ls)
        print(f"long side {ls} ({w}x{h}): {ms:.1f} ms/frame, peak {mem:.0f} MB, " +
              ", ".join(f"{k} {v:.2f}" for k, v in m.items()), flush=True)
        rows += [{"long_side": ls, "input_1920x1080": f"{w}x{h}", "metric": k, "value": round(v, 3)}
                 for k, v in {"ms_per_frame_gpu": ms, "peak_mem_mb": mem, **m}.items()]
        det.close()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache RT-DETR person detections and evaluate detection quality")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--source", choices=["rtdetr_coco", "rtdetr_ft"], required=True)
    c.add_argument("--weights", type=Path, default=None)
    c.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    c.add_argument("--long-side", type=int, default=LONG_SIDE)
    c.add_argument("--batch", type=int, default=4)
    c.add_argument("--workers", type=int, default=6)
    e = sub.add_parser("eval")
    e.add_argument("--splits", nargs="+", choices=EVAL_SPLITS, default=list(EVAL_SPLITS))
    e.add_argument("--sources", nargs="+", default=None)
    e.add_argument("--out", type=Path, default=RESULTS / "detection.csv")
    r = sub.add_parser("resolution")
    r.add_argument("--long-sides", type=int, nargs="+", default=[640, 800, 960, 1088, 1280])
    r.add_argument("--batch", type=int, default=2)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--out", type=Path, default=RESULTS / "detector_resolution.csv")
    args = parser.parse_args()
    if args.cmd == "cache":
        weights = args.weights or (coco_weights() if args.source == "rtdetr_coco" else FT_CHECKPOINT)
        det = Detector.load(weights, long_side=args.long_side)
        cache_source(det, args.source, tuple(args.splits), args.batch, args.workers)
        det.close()
    elif args.cmd == "eval":
        run_eval(args.splits, args.sources, args.out)
    else:
        from mts.train_detector import holdout_frames

        args.out.parent.mkdir(parents=True, exist_ok=True)
        measure_resolution(args.long_sides, holdout_frames(), args.batch, args.workers).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
