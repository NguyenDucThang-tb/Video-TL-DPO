"""CPU checks for validation provenance, scoring, resume, and PBS submission."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from video_tldpo.evaluation.video_validation import (
    PROTOCOL, append_prediction, atomic_json, build_metrics, compare_outputs,
    file_sha256, initialize_output, parse_yes_no, restore_predictions, role_of,
    validate_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("answer, expected", [
    ("Yes.", "yes"), ("  NO!", "no"), ("Yes, there is.", "yes"),
    ("No, there isn't.", "no"), ("Yesterday", None), ("", None),
    ("Yes or no", None), ("No. Yes.", None), ("Maybe yes", None),
])
def test_parser_preserves_invalid_and_ambiguous_answers(answer, expected):
    assert parse_yes_no(answer) == expected


def rows_for(prefix, count, video):
    rows = []
    for index in range(count):
        role = "basic" if index % 2 == 0 else "hallucination"
        subset = "temporal" if (index // 2) % 2 == 0 else "object_relation"
        pair = "{}/{}{}".format(subset, prefix, index // 2)
        rows.append({"sample_id": pair + "|" + role, "pair_id": pair,
                     "subset": subset, "video_path": str(video),
                     "question": "Question {} {}?".format(prefix, index),
                     "gt_answer": "yes" if role == "basic" else "no", "split": prefix})
    return rows


def prediction(row, answer=None):
    answer = row["gt_answer"].capitalize() + "." if answer is None else answer
    parsed = parse_yes_no(answer)
    result = {key: row[key] for key in ("sample_id", "pair_id", "subset", "video_path", "question", "gt_answer")}
    result.update(role=role_of(row), raw_answer=answer, prediction=parsed,
                  correct=parsed == row["gt_answer"], input_digest="input-" + row["sample_id"])
    return result


def fixture_project(tmp_path):
    project = tmp_path / "project"
    artifacts = project / "artifacts"
    artifacts.mkdir(parents=True)
    train_video, val_video = tmp_path / "train.mp4", tmp_path / "val.mp4"
    train_video.write_bytes(b"metadata fixture, not a decoder test")
    val_video.write_bytes(b"metadata fixture, not a decoder test")
    train, val = rows_for("train", 300, train_video), rows_for("val", 100, val_video)
    for name, rows in (("train_binary.jsonl", train), ("val_binary.jsonl", val)):
        (artifacts / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"pair_ids": {"train": sorted({r["pair_id"] for r in train}),
                              "val": sorted({r["pair_id"] for r in val})}}
    (artifacts / "split_manifest.json").write_text(json.dumps(manifest))
    model = project / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "weights.safetensors"}}))
    for name in ("weights.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"):
        (model / name).write_text("fixture")
    training = project / "outputs/vanilla_dpo_qwen25_vh_continuous_v2"
    adapter = training / "trainer_state/latest/adapter"
    adapter.mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (adapter / name).write_text("fixture")
    contract = {"dataset_sha256": file_sha256(artifacts / "train_binary.jsonl"),
                "model_files": {name: file_sha256(model / name) for name in
                                ("config.json", "model.safetensors.index.json", "preprocessor_config.json")}}
    (adapter.parent / "state.json").write_text(json.dumps({"next_record_index": 300, "contract": contract}))
    summary = training / "train_summary.json"
    summary.write_text(json.dumps({"status": "COMPLETED", "records": 300,
                                   "contract": contract, "adapter_checkpoint": str(adapter)}))
    args = (artifacts / "val_binary.jsonl", artifacts / "train_binary.jsonl", model, adapter, summary)
    return project, val, args


def test_preflight_rejects_video_leakage_and_manifest_changes(tmp_path):
    _, val, args = fixture_project(tmp_path)
    rows, info = validate_inputs(*args)
    assert len(rows) == 100 and info["counts"]["video_overlap"] == 0
    train_rows = [json.loads(line) for line in args[1].read_text().splitlines()]
    val[0]["video_path"] = train_rows[0]["video_path"]
    args[0].write_text("".join(json.dumps(row) + "\n" for row in val))
    with pytest.raises(ValueError, match="leakage"):
        validate_inputs(*args)
    val[0]["video_path"] = val[1]["video_path"]
    args[0].write_text("".join(json.dumps(row) + "\n" for row in val))
    manifest = args[0].parent / "split_manifest.json"
    manifest.write_text(json.dumps({"pair_ids": {"train": [], "val": []}}))
    with pytest.raises(ValueError, match="manifest"):
        validate_inputs(*args)


def test_preflight_rejects_changed_training_provenance(tmp_path):
    _, _, args = fixture_project(tmp_path)
    args[1].write_text(args[1].read_text() + "\n")
    with pytest.raises(ValueError, match="provenance"):
        validate_inputs(*args)


def test_resume_recovers_only_an_incomplete_final_line(tmp_path):
    rows = rows_for("val", 2, tmp_path / "v.mp4")
    path = tmp_path / "predictions.jsonl"
    append_prediction(path, prediction(rows[0]))
    with path.open("ab") as handle:
        handle.write(b'{"sample_id":')
    restored = restore_predictions(path, rows)
    assert len(restored) == 1 and path.read_bytes().endswith(b"\n")
    append_prediction(path, prediction(rows[1]))
    assert len(restore_predictions(path, rows)) == 2
    with path.open("ab") as handle:
        handle.write(b"broken\n")
    with pytest.raises(ValueError, match="Corrupt"):
        restore_predictions(path, rows)


def test_resume_rejects_duplicate_rows_or_changed_contract(tmp_path):
    rows = rows_for("val", 2, tmp_path / "v.mp4")
    path = tmp_path / "predictions.jsonl"
    append_prediction(path, prediction(rows[0]))
    append_prediction(path, prediction(rows[0]))
    with pytest.raises(ValueError, match="order/content"):
        restore_predictions(path, rows)
    with pytest.raises(ValueError, match="no provenance"):
        initialize_output(tmp_path, {"protocol": 1})
    atomic_json(tmp_path / "run_meta.json", {"protocol": 1})
    with pytest.raises(ValueError, match="contract changed"):
        initialize_output(tmp_path, {"protocol": 2})


def test_invalid_answers_count_as_wrong_and_pair_accuracy_requires_both():
    rows = rows_for("val", 4, Path("fixture.mp4"))
    predictions = [prediction(rows[0]), prediction(rows[1], "maybe"),
                   prediction(rows[2], "No."), prediction(rows[3])]
    metrics = build_metrics(predictions, expected_count=4)
    assert metrics["overall"]["accuracy"] == 0.5
    assert metrics["overall"]["invalid_answers"] == 1
    assert metrics["overall"]["yes_rate"] == 0.25
    assert metrics["complete_pairs"] == 2 and metrics["pair_accuracy"] == 0


def make_outputs(tmp_path, split="val"):
    count = 100 if split == "val" else 310
    rows = rows_for(split, count, Path("fixture.mp4"))
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    common = {"inputs": {"validation_sha256": "same"}, "protocol": PROTOCOL, "runtime": {"torch": "same"}}
    if split == "test":
        common["inputs"].update(evaluation_split="test", expected_records=310, test_sha256="test")
    for name, directory, wrong in (("base", base, {0, 4}), ("adapter", adapter, {2, 4})):
        atomic_json(directory / "run_meta.json", dict(common, variant=name))
        for index, row in enumerate(rows):
            answer = "No." if index in wrong else None
            append_prediction(directory / "predictions.jsonl", prediction(row, answer))
    return base, adapter


def test_comparison_counts_changes_and_requires_identical_video_inputs(tmp_path):
    base, adapter = make_outputs(tmp_path)
    result = compare_outputs(base, adapter, tmp_path / "comparison.json")
    assert result["base"]["overall"]["accuracy"] == 0.98
    assert result["wrong_to_correct"] == result["correct_to_wrong"] == 1
    assert result["delta_accuracy_pp"] == 0
    path = adapter / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["input_digest"] = "different pixels"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="input_digest"):
        compare_outputs(base, adapter, tmp_path / "comparison.json")


def test_comparison_refuses_incomplete_evaluation(tmp_path):
    base, adapter = make_outputs(tmp_path)
    path = adapter / "predictions.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="all 100"):
        compare_outputs(base, adapter, tmp_path / "comparison.json")


def test_test_comparison_requires_all_310_and_keeps_pair_metrics(tmp_path):
    base, adapter = make_outputs(tmp_path, split="test")
    report = compare_outputs(base, adapter, tmp_path / "comparison.json")
    assert report["split"] == "test" and report["records"] == 310
    assert report["base"]["status"] == "COMPLETED"
    assert report["base"]["complete_pairs"] == 155
    assert report["wrong_to_correct"] == report["correct_to_wrong"] == 1
    path = adapter / "predictions.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:100]) + "\n")
    with pytest.raises(ValueError, match="all 310"):
        compare_outputs(base, adapter, tmp_path / "comparison.json")


def add_test_fixture(project):
    artifacts = project / "artifacts"
    video = project / "heldout-test.mp4"
    video.write_bytes(b"metadata fixture, not decoding")
    rows = rows_for("test", 310, video)
    path = artifacts / "test_records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = artifacts / "split_manifest.json"
    data = json.loads(manifest.read_text())
    data["pair_ids"]["test"] = sorted({r["pair_id"] for r in rows})
    manifest.write_text(json.dumps(data))
    return path, rows


def test_test_preflight_checks_manifest_and_validation_overlap(tmp_path):
    project, _, args = fixture_project(tmp_path)
    path, rows = add_test_fixture(project)
    _, info = validate_inputs(path, *args[1:], split="test")
    assert info["counts"]["test_records"] == 310
    assert info["counts"]["test_pairs"] == 155
    assert info["counts"]["val_test_video_overlap"] == 0
    assert info["expected_records"] == 310 and info["evaluation_split"] == "test"
    rows[0]["video_path"] = json.loads(args[0].read_text().splitlines()[0])["video_path"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="leakage"):
        validate_inputs(path, *args[1:], split="test")


def test_input_digest_detects_changed_pixels():
    spec = importlib.util.spec_from_file_location("evaluate_val", ROOT / "scripts/evaluate_videohallucer_validation.py")
    evaluate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate)
    batch = {"input_ids": torch.tensor([[1, 2]]), "pixel_values_videos": torch.zeros(2, 4),
             "second_per_grid_ts": [0.5]}
    initial = evaluate.tensor_input_digest(batch, torch)
    assert evaluate.tensor_input_digest(batch, torch) == initial
    batch["pixel_values_videos"][0, 0] = 1
    assert evaluate.tensor_input_digest(batch, torch) != initial


@pytest.mark.parametrize("split", ["val", "test"])
def test_validation_alias_routes_to_unified_protocol(tmp_path, split):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    alias = scripts / "submit_videohallucer_validation.sh"
    shutil.copyfile(ROOT / "scripts/submit_videohallucer_validation.sh", alias)
    (scripts / "submit_unified.sh").write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    result = subprocess.run(["bash", str(alias), "--base-only", "--split", split,
                             "--dataset", "videohallucer", "--preflight-only"],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["base", "qwen25", "--split", split,
                                          "--dataset", "videohallucer", "--preflight-only"]
