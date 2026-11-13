import configparser
from pathlib import Path

import numpy as np
import pytest

from mts import evaluate
from mts.common import TRACKEVAL
from tests.test_data import make_seq


def test_mot_round_trip(tmp_path: Path) -> None:
    tracks = np.array([[2, 5, 10.5, 20.25, 30.5, 60.75, 0.875], [1, 3, 0, 0, 4, 8, 1.0], [1, 1, 7, 8, 9, 10, 0.5]])
    path = tmp_path / "out" / "seq.txt"
    evaluate.write_mot(path, tracks)
    lines = path.read_text().splitlines()
    assert lines[0] == "1,1,8.000,9.000,2.000,2.000,0.5000,-1,-1,-1"
    back = evaluate.read_mot(path)
    order = np.lexsort((tracks[:, 1], tracks[:, 0]))
    np.testing.assert_allclose(back, tracks[order], atol=1e-6)


def test_write_mot_without_scores_and_empty(tmp_path: Path) -> None:
    evaluate.write_mot(tmp_path / "a.txt", np.array([[1, 1, 0, 0, 2, 2]]))
    assert evaluate.read_mot(tmp_path / "a.txt")[0, 6] == 1.0
    evaluate.write_mot(tmp_path / "b.txt", np.zeros((0, 7)))
    assert evaluate.read_mot(tmp_path / "b.txt").shape == (0, 7)


def test_crop_raw_gt_renumbers() -> None:
    raw = np.array([[f, 1, 0, 0, 1, 1, 1, 1, 1.0] for f in range(1, 8)])
    val = evaluate.crop_raw_gt(raw, 7, "val_half")
    assert val[:, 0].tolist() == [1, 2, 3, 4]
    train = evaluate.crop_raw_gt(raw, 7, "train_half")
    assert train[:, 0].tolist() == [1, 2, 3]


def test_build_eval_gt_keeps_all_classes_and_rewrites_seqinfo(tmp_path: Path) -> None:
    make_seq(tmp_path / "mot", length=5)
    split_dir = evaluate.build_eval_gt("val_half", tmp_path / "mot", tmp_path / "eval_gt")
    raw = np.loadtxt(split_dir / "MOT17-99" / "gt" / "gt.txt", delimiter=",", ndmin=2)
    assert raw[:, 0].tolist() == [1, 1, 2, 3]
    assert raw[:, 7].tolist() == [1, 7, 1, 1]
    ini = configparser.ConfigParser()
    ini.optionxform = str
    ini.read(split_dir / "MOT17-99" / "seqinfo.ini")
    assert ini["Sequence"]["seqLength"] == "3"
    assert ini["Sequence"]["imWidth"] == "640"
    assert (split_dir / "seqmap.txt").read_text() == "name\nMOT17-99\n"


def test_shuffle_ids_per_frame_keeps_boxes() -> None:
    tracks = np.array([[1, i, i, 0, i + 1, 1, 1] for i in range(5)] + [[2, 9, 0, 0, 1, 1, 1]], dtype=float)
    out = evaluate.shuffle_ids_per_frame(tracks, np.random.default_rng(0))
    np.testing.assert_array_equal(np.delete(out, 1, axis=1), np.delete(tracks, 1, axis=1))
    assert sorted(out[:5, 1]) == list(range(5))
    assert out[5, 1] == 9


@pytest.mark.skipif(not (TRACKEVAL / "trackeval").is_dir(), reason="TrackEval clone missing")
def test_trackeval_perfect_and_swapped_ids(tmp_path: Path) -> None:
    seq_root = tmp_path / "eval_gt" / "val_half" / "S1"
    rows = []
    for f in range(1, 11):
        rows.append([f, 1, 11 + 5 * f, 11, 20, 40, 1, 1, 1.0])
        rows.append([f, 2, 301 - 5 * f, 11, 20, 40, 1, 1, 1.0])
    evaluate.write_raw_gt(seq_root / "gt" / "gt.txt", np.array(rows))
    (seq_root / "seqinfo.ini").write_text("[Sequence]\nname=S1\nseqLength=10\n")
    (tmp_path / "eval_gt" / "val_half" / "seqmap.txt").write_text("name\nS1\n")
    tracks = evaluate.gt_as_tracks("val_half", tmp_path / "eval_gt")["S1"]
    evaluate.write_mot(tmp_path / "trk" / "perfect" / "S1.txt", tracks)
    swapped = tracks.copy()
    swapped[swapped[:, 0] > 5, 1] = 3 - swapped[swapped[:, 0] > 5, 1]
    evaluate.write_mot(tmp_path / "trk" / "swapped" / "S1.txt", swapped)
    perfect = evaluate.run_trackeval(tmp_path / "trk" / "perfect", gt_root=tmp_path / "eval_gt")
    bad = evaluate.run_trackeval(tmp_path / "trk" / "swapped", gt_root=tmp_path / "eval_gt")
    for key in ("HOTA", "MOTA", "IDF1", "DetA", "AssA"):
        assert perfect["S1"][key] == pytest.approx(100.0)
    assert bad["S1"]["DetA"] == pytest.approx(100.0)
    assert bad["S1"]["IDSW"] == 2
    assert bad["S1"]["IDF1"] == pytest.approx(50.0)
    assert set(perfect) == {"S1", evaluate.COMBINED}
