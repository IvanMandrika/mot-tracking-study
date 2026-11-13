import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mts.common import DATA, RESULTS
from mts.data import frame_paths, list_sequences, read_seqinfo
from mts.detector import FT_CHECKPOINT, LONG_SIDE, coco_weights, input_size, load_model, person_detections, \
    person_label, read_rgb, to_tensor
from mts.tracker.matching import iou_matrix

ONNX_DIR = DATA / "deploy" / "onnx"
OPSET = 17
OUTPUTS = ("logits", "pred_boxes")
VARIANTS = {"ort_cuda_fp32": ("fp32", "cuda", "EXHAUSTIVE"), "ort_cuda_fp16": ("fp16", "cuda", "DEFAULT"),
            "ort_cuda_fp16_exhaustive": ("fp16", "cuda", "EXHAUSTIVE"), "ort_cpu_fp32": ("fp32", "cpu", "-"),
            "ort_cpu_int8": ("int8", "cpu", "-")}
CPU_THREADS = 8
FP16_KEEP_FP32 = ["TopK", "GridSample", "Sigmoid", "Softmax", "LayerNormalization", "ReduceMean", "Pow", "Sqrt",
                  "Log", "Div"]
INT8_OPS = ["Conv"]


class DetrOutputs(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes


def export_onnx(module: torch.nn.Module, example: torch.Tensor, path: Path, output_names: tuple[str, ...] = OUTPUTS,
                opset: int = OPSET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(module.eval(), (example,), str(path), input_names=["pixel_values"],
                          output_names=list(output_names), opset_version=opset, dynamo=False, do_constant_folding=True)
    return path


def preload_cuda() -> None:
    import onnxruntime as ort

    torch.zeros(1)
    ort.preload_dlls(cuda=True, cudnn=True, msvc=False)


def make_session(path: Path, device: str, threads: int = CPU_THREADS, conv_algo: str = "EXHAUSTIVE"):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    if device == "cuda":
        preload_cuda()
        providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": conv_algo}), "CPUExecutionProvider"]
    else:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), opts, providers=providers)
    if device == "cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDAExecutionProvider unavailable for {path.name}: {sess.get_providers()}")
    return sess


def to_fp16(src: Path, dst: Path, keep_fp32: list[str] = FP16_KEEP_FP32) -> Path:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = convert_float_to_float16(onnx.load(str(src)), keep_io_types=True, op_block_list=keep_fp32)
    onnx.save(model, str(dst))
    return dst


class FrameReader:
    def __init__(self, tensors: list[np.ndarray]) -> None:
        self.items = iter([{"pixel_values": t} for t in tensors])

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self.items, None)


def to_int8(src: Path, dst: Path, calibration: list[np.ndarray]) -> Path:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = dst.with_name(dst.stem + "_pre.onnx")
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    quantize_static(str(pre), str(dst), FrameReader(calibration), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    op_types_to_quantize=INT8_OPS, calibrate_method=CalibrationMethod.Percentile,
                    extra_options={"CalibPercentile": 99.99})
    pre.unlink()
    return dst


def model_path(precision: str, size: tuple[int, int], tag: str, root: Path = ONNX_DIR) -> Path:
    return root / tag / f"rtdetr_{precision}_{size[0]}x{size[1]}.onnx"


def split_sizes(long_side: int = LONG_SIDE) -> dict[tuple[int, int], list[str]]:
    sizes: dict[tuple[int, int], list[str]] = {}
    for seq in list_sequences():
        info = read_seqinfo(seq)
        sizes.setdefault(input_size(info.width, info.height, long_side), []).append(seq)
    return sizes


def sample_frames(seqs: list[str], split: str, n: int, size: tuple[int, int], seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    paths = [p for s in seqs for p in frame_paths(s, split)]
    pick = sorted(rng.choice(len(paths), size=min(n, len(paths)), replace=False))
    return [to_tensor(read_rgb(paths[i]), size)[None].numpy() for i in pick]


def raw_outputs_torch(module: torch.nn.Module, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        logits, boxes = module(torch.from_numpy(x))
    return logits.numpy(), boxes.numpy()


def match_detections(ref: np.ndarray, got: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.optimize import linear_sum_assignment

    if len(ref) == 0 or len(got) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    iou = iou_matrix(ref, got)
    r, g = linear_sum_assignment(-iou)
    keep = iou[r, g] >= 0.5
    return r[keep], g[keep]


def agreement(ref: list[tuple[np.ndarray, np.ndarray]], got: list[tuple[np.ndarray, np.ndarray]], person: int,
              size: tuple[int, int], min_score: float = 0.3) -> dict[str, float]:
    n_ref, n_got, n_match, box_d, score_d = 0, 0, 0, [0.0], [0.0]
    for r, g in zip(ref, got):
        dr = person_detections(r[0][0], r[1][0], person, *size, min_score=min_score)
        dg = person_detections(g[0][0], g[1][0], person, *size, min_score=min_score)
        ir, ig = match_detections(dr[:, :4], dg[:, :4])
        n_ref, n_got, n_match = n_ref + len(dr), n_got + len(dg), n_match + len(ir)
        box_d.append(float(np.abs(dr[ir, :4] - dg[ig, :4]).max(initial=0.0)))
        score_d.append(float(np.abs(dr[ir, 4] - dg[ig, 4]).max(initial=0.0)))
    return {"dets_ref": n_ref, "dets_variant": n_got, "unmatched_ref": n_ref - n_match,
            "unmatched_variant": n_got - n_match, "matched_frac_of_ref": n_match / max(n_ref, 1),
            "matched_max_abs_box_px": max(box_d), "matched_max_abs_score": max(score_d)}


def cast_nodes(path: Path) -> int:
    import onnx

    return sum(n.op_type == "Cast" for n in onnx.load(str(path), load_external_data=False).graph.node)


def build(weights: Path | str, tag: str, long_side: int, n_check: int, n_calib: int, seed: int,
          check_only: bool = False, root: Path = ONNX_DIR) -> list[dict]:
    model = load_model(weights, "cpu")
    person = person_label(model.config.id2label)
    module = DetrOutputs(model).eval()
    rows = []
    for size, seqs in split_sizes(long_side).items():
        fp32 = model_path("fp32", size, tag, root)
        if not check_only:
            export_onnx(module, torch.zeros(1, 3, size[1], size[0]), fp32)
            to_fp16(fp32, model_path("fp16", size, tag, root))
            to_int8(fp32, model_path("int8", size, tag, root), sample_frames(seqs, "train_half", n_calib, size, seed))
        check = sample_frames(seqs, "val_half", n_check, size, seed + 1)
        ref = [raw_outputs_torch(module, x) for x in check]
        for variant, (precision, device, conv_algo) in VARIANTS.items():
            row = {"variant": variant, "conv_algo": conv_algo, "size": f"{size[0]}x{size[1]}", "frames": len(check)}
            try:
                path = model_path(precision, size, tag, root)
                row["onnx_cast_nodes"] = cast_nodes(path)
                sess = make_session(path, device, conv_algo=conv_algo)
                got = [tuple(sess.run(list(OUTPUTS), {"pixel_values": x})) for x in check]
                row.update(agreement(ref, got, person, size))
                del sess
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            print(row, flush=True)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RT-DETR to ONNX, build fp16 and INT8 copies, check agreement")
    parser.add_argument("--weights", type=Path, default=None, help="default: fine-tuned checkpoint if present")
    parser.add_argument("--tag", default=None, help="subfolder name, default rtdetr_ft or rtdetr_coco")
    parser.add_argument("--long-side", type=int, default=LONG_SIDE)
    parser.add_argument("--check-frames", type=int, default=4)
    parser.add_argument("--calib-frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-only", action="store_true", help="skip export, only re-check agreement")
    args = parser.parse_args()
    ft = (FT_CHECKPOINT / "model.safetensors").exists()
    weights = args.weights or (FT_CHECKPOINT if ft else coco_weights())
    tag = args.tag or ("rtdetr_ft" if Path(weights).resolve() == FT_CHECKPOINT.resolve() else "rtdetr_coco")
    torch.set_num_threads(CPU_THREADS)
    rows = build(weights, tag, args.long_side, args.check_frames, args.calib_frames, args.seed,
                 args.check_only)
    df = pd.DataFrame(rows).melt(id_vars=["variant", "conv_algo", "size", "frames"], var_name="metric",
                                 value_name="value")
    df.insert(0, "weights", tag)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.dropna(subset=["value"]).to_csv(RESULTS / "deploy_agreement.csv", index=False)


if __name__ == "__main__":
    main()
