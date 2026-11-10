"""EventHallusion metadata/scoring tests; fixtures are not real video decoding."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from video_tldpo.evaluation.eventhallusion import (
    EVENT_PROTOCOL, EXPECTED_VIDEOS, compare_eventhallusion, duration_seconds,
    eventhallusion_metrics, load_eventhallusion, official_binary_parse,
    validate_duration, validate_eventhallusion_inputs,
)
from video_tldpo.evaluation.video_validation import (
    append_prediction, atomic_json, parse_yes_no, restore_predictions,
)
from test_video_validation import ROOT, fixture_project


def event_fixture(tmp_path, counts=None):
    counts = counts or {"entire": 2, "mix": 2, "misleading": 1}
    root = tmp_path / "eventhallusion"
    questions = root / "questions"
    questions.mkdir(parents=True)
    for split, count in counts.items():
        directory = root / "videos" / ("interleave" if split == "mix" else split)
        directory.mkdir(parents=True)
        entries = []
        for number in range(1, count + 1):
            name = str(number) if split == "entire" else "interleave_{:03d}".format(number) if split == "mix" else "misleading" + str(number)
            (directory / (name + ".mp4")).write_bytes(b"metadata only; not a decoding test")
            entry = {"id": "{}_{:03d}".format(split, number), "category": "pet&animal",
                     "length": "00:03", "event_info": {"caption": "not given to the model"},
                     "questions": [{"type": "hallucination", "question": "Does the event happen?",
                                    "answer": "Yes." if split == "mix" and number % 2 else "No."}]}
            if split == "entire" and number == 1:
                entry["questions"].append({"type": "hallucination", "question": "Another event?", "answer": "No."})
            entries.append(entry)
        (questions / (split + "_questions.json")).write_text(json.dumps(entries))
    return root, counts


def test_load_resolves_split_scoped_numeric_and_interleave_names(tmp_path):
    root, counts = event_fixture(tmp_path)
    rows, contract = load_eventhallusion(root, expected_video_counts=counts)
    assert len(rows) == 6 and contract["counts"]["eventhallusion_videos"] == 5
    assert contract["counts"]["missing_videos"] == 0
    assert all("caption" not in row and "event_info" not in row for row in rows)
    mix = next(row for row in rows if row["subset"] == "mix")
    assert "interleave/interleave_001.mp4" in mix["video_path"]
    assert mix["gt_answer"] == "yes"
    assert rows[0]["duration_seconds"] == 3


def test_missing_video_never_silently_reduces_benchmark(tmp_path):
    root, counts = event_fixture(tmp_path)
    (root / "videos/interleave/interleave_002.mp4").unlink()
    with pytest.raises(ValueError, match=r"missing 1 annotated videos.*mix/mix_002"):
        load_eventhallusion(root, expected_video_counts=counts)


def test_duplicate_video_ids_and_annotation_versions_are_rejected(tmp_path):
    root, counts = event_fixture(tmp_path)
    alternate = root / "mix"
    alternate.mkdir()
    (alternate / "mix_001.mp4").write_bytes(b"ambiguous second video")
    with pytest.raises(ValueError, match="Multiple video files"):
        load_eventhallusion(root, expected_video_counts=counts)
    (alternate / "mix_001.mp4").unlink()
    backup = root / "backup"
    backup.mkdir()
    shutil.copyfile(root / "questions/entire_questions.json", backup / "entire_questions.json")
    with pytest.raises(ValueError, match="Expected one entire_questions"):
        load_eventhallusion(root, expected_video_counts=counts)
    rows, _ = load_eventhallusion(root, questions_dir=root / "questions", expected_video_counts=counts)
    assert len(rows) == 6


def test_unsupported_answers_and_wrong_annotation_count_fail(tmp_path):
    root, counts = event_fixture(tmp_path)
    with pytest.raises(ValueError, match="annotation video count"):
        load_eventhallusion(root)
    path = root / "questions/entire_questions.json"
    data = json.loads(path.read_text())
    data[0]["questions"][0]["answer"] = "A description."
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Nonbinary"):
        load_eventhallusion(root, expected_video_counts=counts)


@pytest.mark.parametrize("text, official, strict", [
    ("Yes.", "yes", "yes"), ("No", "no", "no"),
    (" No.", None, "no"), ("Yes or no", "yes", None),
    ("Yesterday", "yes", None), ("unknown", None, None),
])
def test_primary_parser_matches_official_prefix_rule(text, official, strict):
    assert official_binary_parse(text) == official
    assert parse_yes_no(text) == strict


def event_prediction(row, answer):
    parsed = official_binary_parse(answer)
    return dict(row, raw_answer=answer, prediction=parsed, strict_prediction=parse_yes_no(answer),
                correct=parsed == row["gt_answer"], input_digest="input-" + row["sample_id"])


def test_metrics_resume_and_comparison_use_binary_question_count_not_pairs(tmp_path):
    root, counts = event_fixture(tmp_path)
    rows, inputs = load_eventhallusion(root, expected_video_counts=counts)
    predictions = [event_prediction(row, row["gt_answer"].capitalize() + ".") for row in rows]
    predictions[0] = event_prediction(rows[0], " No.")
    metrics = eventhallusion_metrics(predictions, len(rows))
    assert metrics["overall"]["correct"] == 5 and metrics["strict_overall"]["correct"] == 6
    assert "pair_accuracy" not in metrics
    for variant in ("base", "adapter"):
        directory = tmp_path / variant
        directory.mkdir()
        atomic_json(directory / "run_meta.json", {"variant": variant, "inputs": inputs,
                                                  "protocol": EVENT_PROTOCOL, "runtime": {"torch": "fixture"}})
        current = predictions if variant == "base" else [event_prediction(r, r["gt_answer"].capitalize() + ".") for r in rows]
        for prediction in current:
            append_prediction(directory / "predictions.jsonl", prediction)
    restored = restore_predictions(tmp_path / "base/predictions.jsonl", rows, parser=official_binary_parse)
    assert len(restored) == 6
    result = compare_eventhallusion(tmp_path / "base", tmp_path / "adapter", tmp_path / "comparison.json")
    assert result["records"] == 6 and result["wrong_to_correct"] == 1
    assert result["adapter"]["overall"]["accuracy"] == 1
    assert result["delta_accuracy_pp"] == pytest.approx(100 / 6)


def test_duration_checks_native_timing_without_assuming_original_fps():
    assert duration_seconds("01:02") == 62
    assert duration_seconds("00:01:02") == 62
    row = {"video_id": "mix_001", "duration_seconds": 10}
    validate_duration(row, {"video_grid_thw": [[4, 2, 2]], "second_per_grid_ts": [2.5]})
    with pytest.raises(RuntimeError, match="duration mismatch"):
        validate_duration(row, {"video_grid_thw": [[4, 2, 2]], "second_per_grid_ts": [0.25]})


def test_event_alias_forwards_root_and_questions_to_unified_job(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    alias = scripts / "submit_videohallucer_validation.sh"
    shutil.copyfile(ROOT / "scripts/submit_videohallucer_validation.sh", alias)
    (scripts / "submit_unified.sh").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$VIDEO_TLDPO_EVENT_ROOT" "$VIDEO_TLDPO_EVENT_QUESTIONS" "$@"\n')
    result = subprocess.run(["bash", str(alias), "--dataset", "eventhallusion", "--base-only",
                             "--dataset-root", "/fixture/videos", "--questions-dir", "/fixture/questions"],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["/fixture/videos", "/fixture/questions", "base", "qwen25",
                                          "--dataset", "eventhallusion"]
