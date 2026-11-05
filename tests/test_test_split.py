"""Test construction of held-out records against fixed annotation ordinal IDs."""
import importlib.util
import json
from pathlib import Path

import pytest

from video_tldpo.evaluation.video_validation import file_sha256

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_test", ROOT / "scripts/build_videohallucer_test.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def dataset_fixture(tmp_path):
    root = tmp_path / "videohallucer"
    for subset in ("temporal", "object_relation"):
        (root / subset / "videos").mkdir(parents=True)
    for split in ("train", "val", "test"):
        (root / "temporal/videos" / (split + ".mp4")).write_bytes(b"fixture, not decoding")
    items, anchors = [], {"train": [], "val": []}
    ids = {"train": [], "val": [], "test": []}
    for index in range(355):
        split = "train" if index < 150 else "val" if index < 200 else "test"
        pair_id = "temporal/" + str(index)
        ids[split].append(pair_id)
        item = {}
        for role, answer in (("basic", "yes"), ("hallucination", "no")):
            question = role + " " + str(index)
            item[role] = {"video": split + ".mp4", "question": question, "answer": answer}
            if split in anchors:
                anchors[split].append({"sample_id": pair_id + "|" + role, "pair_id": pair_id,
                                       "split": split, "subset": "temporal", "question": question,
                                       "gt_answer": answer, "video_path": str(root / "temporal/videos" / (split + ".mp4"))})
        items.append(item)
    # Same duplicate handling as original split creation must preserve ordinal IDs.
    items.insert(6, items[5])
    annotation = root / "temporal/temporal.json"
    annotation.write_text(json.dumps(items))
    ids["test"].reverse()  # the manifest order must be preserved, not recreated/sorted
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(json.dumps({"pair_ids": ids}))
    for split, rows in anchors.items():
        (tmp_path / (split + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "test_records.jsonl"
    return (root, manifest, tmp_path / "train.jsonl", tmp_path / "val.jsonl", output), annotation


def test_build_preserves_ids_order_sources_and_existing_output(tmp_path):
    args, _ = dataset_fixture(tmp_path)
    before = [file_sha256(path) for path in args[1:4]]
    rows = builder.build_test(*args)
    assert len(rows) == 310 and len({r["pair_id"] for r in rows}) == 155
    assert rows[0]["sample_id"] == "temporal/354|basic"
    assert all(r["split"] == "test" and "chosen" not in r and "rejected" not in r for r in rows)
    assert [file_sha256(path) for path in args[1:4]] == before
    digest = file_sha256(args[-1])
    builder.build_test(*args)
    assert file_sha256(args[-1]) == digest
    args[-1].write_text("{}\n")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        builder.build_test(*args)


def test_build_rejects_changed_annotation_ids_before_writing(tmp_path):
    args, _ = dataset_fixture(tmp_path)
    rows = [json.loads(line) for line in args[2].read_text().splitlines()]
    rows[0]["question"] = "changed anchor"
    args[2].write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="anchor"):
        builder.build_test(*args)
    assert not args[-1].exists()


def test_build_rejects_test_video_leakage(tmp_path):
    args, annotation = dataset_fixture(tmp_path)
    data = json.loads(annotation.read_text())
    for role in ("basic", "hallucination"):
        data[201][role]["video"] = "train.mp4"
    annotation.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="leakage"):
        builder.build_test(*args)
    assert not args[-1].exists()
