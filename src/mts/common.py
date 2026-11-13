import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "cache"
RESULTS = Path(os.environ.get("MTS_RESULTS", ROOT / "results"))
FIGURES = Path(os.environ.get("MTS_FIGURES", ROOT / "figures"))
MOT17 = DATA / "MOT17"
TRACKEVAL = DATA / "TrackEval"
EVAL_GT = DATA / "eval_gt"
SPLITS = ("train_half", "val_half")
SUBSPLITS = ("train_fit", "train_holdout")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
