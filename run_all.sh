#!/usr/bin/env bash
# Reproduces every table and figure from a clean clone. ALLOW_CPU=1 runs only the CPU-feasible subset.
set -euo pipefail
cd "$(dirname "$0")"

MOT17_URL="https://motchallenge.net/data/MOT17.zip"
TRACKEVAL_URL="https://github.com/JonathonLuiten/TrackEval"
TRACKEVAL_COMMIT="12c8791b303e0a0b50f753af204249e622d0281a"
PY=.venv/bin/python
ALLOW_CPU="${ALLOW_CPU:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p data/logs
exec > >(tee -a "data/logs/run_all_$(date +%Y%m%d_%H%M%S).log") 2>&1
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

if [ ! -x "$PY" ]; then
    step "venv"
    python3.12 -m venv .venv
fi
step "install pinned requirements"
.venv/bin/pip install -q -r requirements.lock
.venv/bin/pip install -q --no-deps -e .

if "$PY" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    GPU=1
    DET_SOURCES="public_frcnn rtdetr_coco rtdetr_ft"
elif [ "$ALLOW_CPU" = "1" ]; then
    GPU=0
    DET_SOURCES="public_frcnn"
    export MTS_RESULTS="$PWD/results/cpu_subset" MTS_FIGURES="$PWD/figures/cpu_subset"
    echo "CUDA unavailable: ALLOW_CPU=1, running the CPU subset (no RT-DETR caching, fine-tuning or deployment)"
else
    echo "CUDA unavailable: refusing to run. Set ALLOW_CPU=1 to run the CPU subset only." >&2
    exit 1
fi

if [ ! -d data/MOT17/train ]; then
    step "download MOT17 (resumable)"
    if command -v aria2c > /dev/null; then
        aria2c -c -x 8 -s 8 -d data -o MOT17.zip "$MOT17_URL"
    else
        wget -c -O data/MOT17.zip "$MOT17_URL"
    fi
    unzip -t -q data/MOT17.zip
    unzip -q -n data/MOT17.zip -d data/
fi

if [ ! -d data/TrackEval/.git ]; then
    step "clone TrackEval"
    git clone -q "$TRACKEVAL_URL" data/TrackEval
fi
git -C data/TrackEval checkout -q "$TRACKEVAL_COMMIT"

step "tests"
"$PY" -m pytest -q

step "half-split GT, public detections, evaluation sanity"
"$PY" -m mts.data
"$PY" -m mts.evaluate build-gt
"$PY" -m mts.evaluate sanity --split val_half
"$PY" -m mts.evaluate sanity --split train_half
"$PY" -m mts.evaluate sanity --split train_holdout

if [ "$GPU" = "1" ]; then
    step "detector: resolution sweep on held-out TRAIN-HALF, COCO detections"
    "$PY" -m mts.detect resolution
    "$PY" -m mts.detect cache --source rtdetr_coco
    step "detector: fine-tuning on TRAIN-HALF"
    "$PY" -m mts.train_detector
    "$PY" -m mts.detect cache --source rtdetr_ft
fi
"$PY" -m mts.detect eval --sources $DET_SOURCES

step "tracker caches: camera motion and ReID embeddings"
"$PY" -m mts.tracker.cmc
if [ "$GPU" = "1" ]; then
    "$PY" -m mts.tracker.reid --sources rtdetr_ft public_frcnn rtdetr_coco --device cuda --batch 64
else
    "$PY" -m mts.tracker.reid --sources public_frcnn --device cpu --threads 6
fi

step "tracking ablation, oracle sanity, demo"
"$PY" -m mts.exp_tracking --workers 4
"$PY" -m mts.exp_tracking --oracle
"$PY" -m mts.demo

if [ "$GPU" = "1" ]; then
    step "deployment: ONNX export, ORT variants, latency and quality"
    "$PY" -m mts.deploy.export --check-frames 8 --calib-frames 64
    "$PY" -m mts.deploy.bench
fi

step "done"
ls -l "${MTS_RESULTS:-results}" "${MTS_FIGURES:-figures}"
