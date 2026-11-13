from pathlib import Path

import numpy as np
import pytest

from mts import data

GT_ROWS = [
    "1,1,11,21,10,20,1,1,1.0",
    "1,2,51,61,5,5,0,1,1.0",
    "2,1,12,22,10,20,1,1,0.2",
    "3,1,13,23,10,20,1,1,0.25",
    "3,3,100,100,8,8,1,7,1.0",
    "4,4,1,1,4,4,1,1,0.9",
    "5,4,2,2,4,4,1,1,0.9",
]
DET_ROWS = ["1,-1,11,21,10,20,0.9,-1,-1,-1", "3,-1,5,5,5,5,0.01,-1,-1,-1", "4,-1,1,1,2,2,0.5,-1,-1,-1"]


def make_seq(root: Path, seq: str = "MOT17-99", length: int = 5) -> None:
    for copy in ("FRCNN", "DPM"):
        d = root / "train" / f"{seq}-{copy}"
        (d / "gt").mkdir(parents=True)
        (d / "det").mkdir()
        (d / "seqinfo.ini").write_text(
            f"[Sequence]\nname={seq}-{copy}\nimDir=img1\nframeRate=30\nseqLength={length}\n"
            "imWidth=640\nimHeight=480\nimExt=.jpg\n"
        )
        (d / "gt" / "gt.txt").write_text("\n".join(GT_ROWS) + "\n")
        (d / "det" / "det.txt").write_text("\n".join(DET_ROWS) + "\n")


@pytest.mark.parametrize(
    "length,train,val",
    [(10, (1, 5), (6, 10)), (11, (1, 5), (6, 11)), (1, (1, 0), (1, 1)), (600, (1, 300), (301, 600)), (525, (1, 262), (263, 525))],
)
def test_half_range(length: int, train: tuple[int, int], val: tuple[int, int]) -> None:
    assert data.half_range(length, "train_half") == train
    assert data.half_range(length, "val_half") == val
    assert data.split_length(length, "train_half") + data.split_length(length, "val_half") == length


def test_half_range_rejects_unknown_split() -> None:
    with pytest.raises(ValueError):
        data.half_range(10, "test")


def test_select_split_frames_renumbers_from_one() -> None:
    frames = np.array([1, 2, 3, 4, 5, 5, 6, 7])
    mask, train = data.select_split_frames(frames, 7, "train_half")
    assert train.tolist() == [1, 2, 3]
    mask, val = data.select_split_frames(frames, 7, "val_half")
    assert frames[mask].tolist() == [4, 5, 5, 6, 7]
    assert val.tolist() == [1, 2, 2, 3, 4]


def test_sequence_listing_and_seqinfo(tmp_path: Path) -> None:
    make_seq(tmp_path)
    make_seq(tmp_path, "MOT17-01", 4)
    assert data.list_sequences(tmp_path) == ["MOT17-01", "MOT17-99"]
    info = data.read_seqinfo("MOT17-99", tmp_path)
    assert (info.length, info.width, info.height, info.fps) == (5, 640, 480, 30.0)
    paths = data.frame_paths("MOT17-99", "val_half", tmp_path)
    assert [p.name for p in paths] == ["000003.jpg", "000004.jpg", "000005.jpg"]
    assert paths[0].parent == tmp_path / "train" / "MOT17-99-FRCNN" / "img1"


def test_parse_gt_converts_to_zero_based_corners(tmp_path: Path) -> None:
    make_seq(tmp_path)
    gt = data.load_gt("MOT17-99", root=tmp_path)
    assert list(gt.columns) == data.GT_COLUMNS
    first = gt.iloc[0]
    assert (first.x1, first.y1, first.x2, first.y2) == (10, 20, 20, 40)


def test_gt_filter_class_consider_visibility(tmp_path: Path) -> None:
    make_seq(tmp_path)
    gt = data.filter_gt(data.load_gt("MOT17-99", root=tmp_path))
    assert list(zip(gt.frame, gt.id)) == [(1, 1), (3, 1), (4, 4), (5, 4)]
    assert len(data.filter_gt(data.load_gt("MOT17-99", root=tmp_path), min_vis=0.0)) == 5


def test_gt_split_crop(tmp_path: Path) -> None:
    make_seq(tmp_path)
    val = data.load_training_gt("MOT17-99", "val_half", root=tmp_path)
    assert list(zip(val.frame, val.id)) == [(1, 1), (2, 4), (3, 4)]
    train = data.load_gt("MOT17-99", "train_half", root=tmp_path)
    assert train.frame.max() == 2 and len(train) == 3


def test_public_dets_split_and_score_filter(tmp_path: Path) -> None:
    make_seq(tmp_path)
    val = data.load_public_dets("MOT17-99", "val_half", root=tmp_path)
    assert val.dtype == np.float32 and val.shape == (1, 6)
    np.testing.assert_allclose(val[0], [2, 0, 0, 2, 2, 0.5])
    train = data.load_public_dets("MOT17-99", "train_half", root=tmp_path)
    np.testing.assert_allclose(train[0], [1, 10, 20, 20, 40, 0.9])


def test_dets_cache_round_trip(tmp_path: Path) -> None:
    dets = np.array([[1, 0, 0, 5, 5, 0.9], [3, 1, 1, 4, 4, 0.2]], dtype=np.float64)
    path = data.save_dets(dets, "src", "val_half", "MOT17-99", tmp_path)
    assert path == tmp_path / "dets" / "src" / "val_half" / "MOT17-99.npy"
    loaded = data.load_dets("src", "val_half", "MOT17-99", tmp_path)
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded, dets, rtol=1e-6)
    with pytest.raises(ValueError):
        data.save_dets(dets[:, :5], "src", "val_half", "x", tmp_path)


def test_dets_by_frame_groups_and_keeps_empty_frames() -> None:
    dets = np.array([[3, 1, 1, 2, 2, 0.5], [1, 0, 0, 1, 1, 0.9], [3, 2, 2, 3, 3, 0.4]], dtype=np.float32)
    groups = data.dets_by_frame(dets, 4)
    assert [len(g) for g in groups] == [1, 0, 2, 0]
    assert groups[2].shape == (2, 5)
    np.testing.assert_allclose(groups[0][0], [0, 0, 1, 1, 0.9])
