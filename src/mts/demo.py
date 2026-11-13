import argparse
import colorsys
import json

import cv2
import numpy as np
from PIL import Image

from mts.common import FIGURES, RESULTS
from mts.data import frame_paths, read_seqinfo
from mts.exp_tracking import load_inputs
from mts.tracker.tracker import TrackerConfig, track_sequence

MAX_BYTES = 5 * 1024 * 1024


def id_colour(track_id: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((track_id * 0.618034) % 1.0, 0.85, 0.95)
    return int(255 * r), int(255 * g), int(255 * b)


def draw(img: np.ndarray, tracks: np.ndarray, scale: float) -> np.ndarray:
    for _, tid, x1, y1, x2, y2, _ in tracks:
        c = id_colour(int(tid))
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(img, p1, p2, c, 2)
        label = str(int(tid))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 4), (p1[0] + tw + 4, p1[1]), c, -1)
        cv2.putText(img, label, (p1[0] + 2, p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def render(seq: str, source: str, cfg: TrackerConfig, seconds: float, stride: int, width: int) -> list[Image.Image]:
    inp = load_inputs(source, "val_half", seq, cfg.reid)
    tracks, _ = track_sequence(cfg, inp["frames"], inp["fps"], inp["feats"], inp["affines"])
    n = min(len(inp["frames"]), int(round(seconds * inp["fps"])))
    images = []
    for t, path in enumerate(frame_paths(seq, "val_half")[:n:stride]):
        f = t * stride + 1
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        scale = width / img.shape[1]
        img = cv2.resize(img, (width, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        images.append(Image.fromarray(draw(img, tracks[tracks[:, 0] == f], scale)))
    return images


def save_gif(images: list[Image.Image], fps: float, stride: int, colours: int) -> int:
    palette = images[len(images) // 2].quantize(colors=colours, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    path = FIGURES / "demo.gif"
    FIGURES.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(round(1000 * stride / fps)), loop=0,
                   optimize=True)
    return path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a short GIF of the best tracker on one VAL-HALF sequence")
    parser.add_argument("--seq", default="MOT17-04")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--width", type=int, default=400)
    args = parser.parse_args()
    best = json.loads((RESULTS / "tracking_best.json").read_text())
    cfg = TrackerConfig(**best["config"])
    images = render(args.seq, best["source"], cfg, args.seconds, args.stride, args.width)
    fps = read_seqinfo(args.seq).fps
    for colours in (128, 64):
        size = save_gif(images, fps, args.stride, colours)
        if size < MAX_BYTES:
            break
    print(f"{args.seq} {best['source']} {best['variant']}: {len(images)} frames, {size / 1e6:.2f} MB, {colours} colours")


if __name__ == "__main__":
    main()
