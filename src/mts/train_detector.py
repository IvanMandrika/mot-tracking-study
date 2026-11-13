import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from mts.common import RESULTS, seed_everything
from mts.data import HOLDOUT_FRACTION, MIN_VISIBILITY, frame_paths, half_range, list_sequences, load_training_gt, \
    read_seqinfo
from mts.detector import FT_CHECKPOINT, LONG_SIDE, Detector, coco_weights, input_size, load_model, \
    person_label, read_rgb, worker_init, xyxy_to_cxcywh_norm

PAD_VALUE = 114
MIN_KEPT_AREA = 0.4
MIN_BOX_PX = 2.0


def subsplit_frames(split: str) -> dict[str, list[int]]:
    out = {}
    for s in list_sequences():
        first, last = half_range(read_seqinfo(s).length, split)
        out[s] = list(range(first, last + 1))
    return out


def train_frames() -> dict[str, list[int]]:
    return subsplit_frames("train_fit")


def holdout_frames() -> dict[str, list[int]]:
    return subsplit_frames("train_holdout")


def scale_jitter_affine(width: int, height: int, out_size: tuple[int, int], zoom: float, u: float,
                        v: float) -> np.ndarray:
    win_w, win_h = width / zoom, height / zoom
    x0 = u * (width - win_w)
    y0 = v * (height - win_h)
    sx, sy = out_size[0] / win_w, out_size[1] / win_h
    return np.array([[sx, 0.0, -x0 * sx], [0.0, sy, -y0 * sy]])


def transform_boxes(boxes: np.ndarray, m: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    b = boxes * np.array([m[0, 0], m[1, 1], m[0, 0], m[1, 1]]) + np.array([m[0, 2], m[1, 2], m[0, 2], m[1, 2]])
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    c = b.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(0, out_size[0])
    c[:, [1, 3]] = c[:, [1, 3]].clip(0, out_size[1])
    w, h = c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]
    keep = (w >= MIN_BOX_PX) & (h >= MIN_BOX_PX) & (w * h >= MIN_KEPT_AREA * area)
    return c[keep]


def colour_jitter(rgb: np.ndarray, rng: np.random.Generator, hue: float = 0.015, sat: float = 0.4,
                  val: float = 0.4) -> np.ndarray:
    gains = rng.uniform(-1, 1, 3) * np.array([hue, sat, val]) + np.array([0.0, 1.0, 1.0])
    h, s, v = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
    x = np.arange(256, dtype=np.float64)
    lut_h = ((x + gains[0] * 180) % 180).astype(np.uint8)
    lut_s = np.clip(x * gains[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * gains[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(rgb: np.ndarray, boxes: np.ndarray, out_size: tuple[int, int], rng: np.random.Generator,
            zoom_range: tuple[float, float] = (0.75, 1.5)) -> tuple[np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    zoom = math.exp(rng.uniform(math.log(zoom_range[0]), math.log(zoom_range[1])))
    m = scale_jitter_affine(width, height, out_size, zoom, rng.uniform(), rng.uniform())
    img = cv2.warpAffine(rgb, m, out_size, flags=cv2.INTER_LINEAR, borderValue=(PAD_VALUE,) * 3)
    boxes = transform_boxes(boxes, m, out_size)
    if rng.uniform() < 0.5:
        img = img[:, ::-1]
        boxes = np.stack([out_size[0] - boxes[:, 2], boxes[:, 1], out_size[0] - boxes[:, 0], boxes[:, 3]], axis=1)
    return colour_jitter(np.ascontiguousarray(img), rng), boxes


class TrainSet(Dataset):
    def __init__(self, frames: dict[str, list[int]], long_side: int, seed: int) -> None:
        self.items, self.sizes = [], []
        for seq, fr in frames.items():
            info = read_seqinfo(seq)
            gt = load_training_gt(seq, "train_half")
            groups = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in gt.groupby("frame")}
            paths = frame_paths(seq, "train_half")
            size = input_size(info.width, info.height, long_side)
            for f in fr:
                self.items.append((paths[f - 1], groups.get(f, np.zeros((0, 4)))))
                self.sizes.append(size)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, np.ndarray]:
        rng = np.random.default_rng((self.seed, self.epoch, i))
        path, boxes = self.items[i]
        size = self.sizes[i]
        img, boxes = augment(read_rgb(path), boxes, size, rng)
        x = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        return x, xyxy_to_cxcywh_norm(boxes, *size).astype(np.float32)


class SizeGroupedBatches(Sampler):
    def __init__(self, sizes: list[tuple[int, int]], batch: int, seed: int) -> None:
        self.sizes = sizes
        self.batch = batch
        self.seed = seed
        self.epoch = 0

    def batches(self) -> list[list[int]]:
        rng = np.random.default_rng((self.seed, self.epoch))
        order = rng.permutation(len(self.sizes))
        groups: dict[tuple[int, int], list[int]] = {}
        for i in order:
            groups.setdefault(self.sizes[i], []).append(int(i))
        out = [g[k:k + self.batch] for g in groups.values() for k in range(0, len(g), self.batch)]
        return [out[j] for j in rng.permutation(len(out))]

    def __iter__(self):
        return iter(self.batches())

    def __len__(self) -> int:
        return len(self.batches())


def collate(batch: list) -> tuple[torch.Tensor, list[np.ndarray]]:
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def make_targets(boxes: list[np.ndarray], person: int, device: str) -> list[dict[str, torch.Tensor]]:
    return [{"class_labels": torch.full((len(b),), person, dtype=torch.long, device=device),
             "boxes": torch.from_numpy(b).to(device)} for b in boxes]


def param_groups(model, lr: float, backbone_lr: float, weight_decay: float) -> list[dict]:
    backbone, rest = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (backbone if ".backbone." in f".{name}" else rest).append(p)
    return [{"params": backbone, "lr": backbone_lr}, {"params": rest, "lr": lr}]


def lr_factor(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def holdout_ap(model, long_side: int, frames: dict[str, list[int]], batch: int, workers: int) -> dict[str, float]:
    from mts.detect import coco_metrics, detect_frames, split_targets
    from mts.data import dets_by_frame

    model.eval()
    det = Detector(model, long_side=long_side)
    gts, igns = split_targets("train_half", frames)
    dets = []
    for seq, fr in frames.items():
        d = detect_frames(det, seq, fr, "train_half", batch, workers)
        d[:, 0] -= fr[0] - 1
        dets += dets_by_frame(d, len(fr))
    model.train()
    return coco_metrics(gts, igns, dets)


def train_step(model, optimizer, x: torch.Tensor, targets: list[dict], amp: bool, scaler, clip: float,
               chunks: int = 1) -> float:
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for xs, ts in zip(x.chunk(chunks), np.array_split(np.arange(len(targets)), chunks)):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = model(pixel_values=xs, labels=[targets[i] for i in ts]).loss * len(ts) / len(targets)
        scaler.scale(loss).backward()
        total += float(loss.detach())
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    scaler.step(optimizer)
    scaler.update()
    return total


def safe_train_step(model, optimizer, x: torch.Tensor, targets: list[dict], amp: bool, scaler, clip: float) -> float:
    chunks = 1
    while True:
        try:
            return train_step(model, optimizer, x, targets, amp, scaler, clip, chunks)
        except torch.cuda.OutOfMemoryError:
            if chunks >= len(targets):
                raise
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            chunks *= 2
            print(f"CUDA OOM, waiting 30 s and retrying in {chunks} chunks", flush=True)
            time.sleep(30)


def plot_training(log: pd.DataFrame, path_name: str = "detector_training.png") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from mts.plotting import SERIES, save, setup

    setup()
    wide = log.pivot_table(index="epoch", columns="metric", values="value")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    tr = wide["train_loss"].dropna()
    a.plot(tr.index, tr.values, marker="o", color=SERIES[0])
    a.set_title("Training loss per epoch")
    a.set_xlabel("epoch")
    a.set_ylabel("mean RT-DETR loss")
    for col, colour, label in (("holdout_AP", SERIES[1], "AP@[.5:.95]"), ("holdout_AP50", SERIES[2], "AP50")):
        s = wide[col].dropna()
        b.plot(s.index, s.values, marker="o", color=colour, label=label)
    b.set_title("Held-out TRAIN-HALF AP (epoch 0 = COCO weights)")
    b.set_xlabel("epoch")
    b.set_ylabel("AP, %")
    b.legend()
    for ax in (a, b):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    save(fig, path_name)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = "cuda"
    model = load_model(coco_weights(), device).train()
    person = person_label(model.config.id2label)
    ds = TrainSet(train_frames(), args.long_side, args.seed)
    sampler = SizeGroupedBatches(ds.sizes, args.batch, args.seed)
    hold = holdout_frames()
    steps_per_epoch = len(sampler)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.backbone_lr, args.weight_decay),
                            weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    rows = []
    m0 = holdout_ap(model, args.long_side, hold, args.eval_batch, args.workers)
    rows += [{"epoch": 0, "metric": f"holdout_{k}", "value": v} for k, v in m0.items()]
    print(f"epoch 0 (COCO weights): {m0}", flush=True)
    best, bad, step = m0["AP"], 0, 0
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        ds.epoch = sampler.epoch = epoch
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=args.workers,
                            pin_memory=True, worker_init_fn=worker_init)
        t0 = time.perf_counter()
        losses = []
        for x, boxes in loader:
            f = lr_factor(step, args.warmup, total)
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * f
            losses.append(safe_train_step(model, opt, x.to(device, non_blocking=True), make_targets(boxes, person, device),
                                     args.amp, scaler, args.clip))
            step += 1
        train_s = time.perf_counter() - t0
        m = holdout_ap(model, args.long_side, hold, args.eval_batch, args.workers)
        epoch_s = time.perf_counter() - t0
        rows += [{"epoch": epoch, "metric": "train_loss", "value": float(np.mean(losses))},
                 {"epoch": epoch, "metric": "train_seconds", "value": train_s},
                 {"epoch": epoch, "metric": "epoch_seconds", "value": epoch_s}]
        rows += [{"epoch": epoch, "metric": f"holdout_{k}", "value": v} for k, v in m.items()]
        improved = m["AP"] > best
        print(f"epoch {epoch}: loss {np.mean(losses):.4f}, holdout {m}, {train_s:.0f}+{epoch_s - train_s:.0f} s"
              + (" (best)" if improved else ""), flush=True)
        if improved:
            best, bad = m["AP"], 0
            args.out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            (args.out / "train_info.json").write_text(json.dumps(
                {"epoch": epoch, "holdout": m, "seed": args.seed, "long_side": args.long_side,
                 "min_visibility": MIN_VISIBILITY, "holdout_fraction": HOLDOUT_FRACTION,
                 "args": {k: str(v) for k, v in vars(args).items()}}, indent=1))
        else:
            bad += 1
        pd.DataFrame(rows).to_csv(args.log, index=False)
        elapsed = time.perf_counter() - t_start
        if bad >= args.patience:
            print(f"early stop: no held-out AP gain for {args.patience} epochs", flush=True)
            break
        if args.budget_min and elapsed + epoch_s > args.budget_min * 60:
            print(f"stopping: next epoch would exceed the {args.budget_min} min budget", flush=True)
            break
    rows.append({"epoch": epoch, "metric": "total_minutes", "value": (time.perf_counter() - t_start) / 60})
    pd.DataFrame(rows).to_csv(args.log, index=False)
    plot_training(pd.DataFrame(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RT-DETR-R18 on MOT17 TRAIN-HALF pedestrians")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--budget-min", type=float, default=0.0, help="optional wall-clock stop, 0 = off")
    parser.add_argument("--long-side", type=int, default=LONG_SIDE)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FT_CHECKPOINT)
    parser.add_argument("--log", type=Path, default=RESULTS / "detector_training.csv")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
