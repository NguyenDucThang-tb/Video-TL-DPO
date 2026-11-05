"""Offline EventHallusion binary QA loading, coverage checks, and metrics.

The primary parser mirrors Stevetich/EventHallusion eval_binary.py: lowercase
then startswith('yes'/'no'). The stricter VH parser is reported separately.
No open-ended descriptions or remote judging services are used.
"""
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re

from .video_validation import (
    PROTOCOL, atomic_json, file_sha256, parse_yes_no, question_summary,
    read_jsonl, trained_adapter_contract, validate_rows,
)

EXPECTED_VIDEOS = {"entire": 109, "mix": 193, "misleading": 95}
SPLIT_NAMES = {"entire": "entire", "mix": "mix", "interleave": "mix", "misleading": "misleading"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}
EVENT_PROTOCOL = dict(PROTOCOL, parser="eventhallusion_eval_binary_startswith_v1",
                      strict_parser_diagnostic=PROTOCOL["parser"], task="binary_question_accuracy",
                      duration_check="diagnostic_only_abs_difference_vs_max(1.5s,10pct_annotated)")


def official_binary_parse(text):
    lowered = text.lower()
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    return None


def duration_seconds(value):
    if value is None or value == "":
        return None
    parts = str(value).strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError("Invalid annotated duration: " + str(value))
    seconds = sum(float(part) * (60 ** index) for index, part in enumerate(reversed(parts)))
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid annotated duration: " + str(value))
    return seconds


def validate_duration(row, native_inputs, strict=True):
    """Compare annotation duration with decoded clip duration; optionally enforce it."""
    if row.get("duration_seconds") is None:
        return None
    observed = float(native_inputs["video_grid_thw"][0][0]) * float(native_inputs["second_per_grid_ts"][0])
    expected = row["duration_seconds"]
    difference = abs(observed - expected)
    tolerance = max(1.5, 0.1 * expected)
    result = {"annotated_seconds": expected, "decoded_seconds": observed,
              "absolute_difference_seconds": difference, "tolerance_seconds": tolerance,
              "within_tolerance": difference <= tolerance}
    if strict and not result["within_tolerance"]:
        raise RuntimeError("EventHallusion ID/video duration mismatch: {} annotated={} observed={}".format(
            row["video_id"], expected, observed))
    return result


def walk_files(root):
    visited = set()
    for current, directories, files in os.walk(str(root), followlinks=True):
        real = str(Path(current).resolve())
        if real in visited:
            directories[:] = []
            continue
        visited.add(real)
        directories[:] = sorted(d for d in directories if d != "__MACOSX")
        for name in sorted(files):
            yield Path(current) / name


def numeric_id(stem):
    match = re.fullmatch(r"(?:(entire|mix|interleave|misleading)[_-]?)?0*(\d+)", stem.lower())
    if not match:
        return None, None
    return SPLIT_NAMES.get(match.group(1)), int(match.group(2))


def load_eventhallusion(root, questions_dir=None, expected_video_counts=None):
    root = Path(root)
    if not root.is_dir():
        raise ValueError("EventHallusion directory does not exist: " + str(root))
    expected = EXPECTED_VIDEOS if expected_video_counts is None else expected_video_counts
    files = list(walk_files(root))
    annotations = {}
    for split in ("entire", "mix", "misleading"):
        name = split + "_questions.json"
        candidates = [Path(questions_dir) / name] if questions_dir else [p for p in files if p.name == name]
        candidates = list({str(p.resolve()): p for p in candidates if p.is_file()}.values())
        if len(candidates) != 1:
            raise ValueError("Expected one {} (found {}); use --questions-dir for the intended version".format(name, len(candidates)))
        annotations[split] = candidates[0]

    index, unmatched = defaultdict(list), []
    for path in files:
        if path.suffix.lower() not in VIDEO_EXTENSIONS or "__MACOSX" in path.parts:
            continue
        prefix, number = numeric_id(path.stem)
        folders = {SPLIT_NAMES[part.lower()] for part in path.relative_to(root).parts[:-1] if part.lower() in SPLIT_NAMES}
        if len(folders) > 1:
            raise ValueError("Ambiguous split folders for " + str(path))
        folder = next(iter(folders)) if folders else None
        if prefix and folder and prefix != folder:
            raise ValueError("Filename and split directory disagree: " + str(path))
        split = prefix or folder
        if split is None or number is None:
            unmatched.append(str(path))
            continue
        key = (split, number)
        if str(path.resolve()) not in {str(p.resolve()) for p in index[key]}:
            index[key].append(path)

    rows, counts, missing, used_files = [], {}, [], set()
    for split, annotation in annotations.items():
        entries = json.loads(annotation.read_text())
        if not isinstance(entries, list) or len(entries) != expected[split]:
            raise ValueError("{} annotation video count: expected {}, found {}; check annotation version".format(
                split, expected[split], len(entries) if isinstance(entries, list) else type(entries).__name__))
        seen_ids, seen_numbers, n_questions = set(), set(), 0
        for entry in entries:
            video_id = entry["id"]
            prefix, number = numeric_id(video_id)
            if prefix != split or number is None or number <= 0 or video_id in seen_ids or number in seen_numbers:
                raise ValueError("Invalid/duplicate EventHallusion ID: " + str(video_id))
            seen_ids.add(video_id)
            seen_numbers.add(number)
            matches = index.get((split, number), [])
            if len(matches) > 1:
                raise ValueError("Multiple video files map to {}: {}".format(video_id, [str(p) for p in matches]))
            if not matches:
                missing.append(split + "/" + video_id)
                continue
            path = matches[0]
            used_files.add(str(path.resolve()))
            questions = entry["questions"]
            if not isinstance(questions, list) or not questions:
                raise ValueError("Video has no binary questions: " + video_id)
            category = entry.get("category")
            if not isinstance(category, str) or not category.strip():
                raise ValueError("Missing category for " + video_id)
            for qi, question in enumerate(questions):
                answer = str(question["answer"]).strip().rstrip(".").lower()
                if answer not in ("yes", "no") or not question.get("question"):
                    raise ValueError("Nonbinary/empty question: {} index {}".format(video_id, qi))
                rows.append({"sample_id": "eh/{}/{}|{}".format(split, video_id, qi),
                             "pair_id": "eh/{}/{}/{}".format(split, video_id, qi),
                             "source_dataset": "EventHallusion", "subset": split, "role": "eventhallusion",
                             "video_id": video_id, "category": category,
                             "question_type": question.get("type", "binary"),
                             "duration_seconds": duration_seconds(entry.get("length")),
                             "video_path": str(path), "question": question["question"], "gt_answer": answer})
                n_questions += 1
        counts[split] = {"annotation_videos": len(entries), "matched_videos": len(seen_ids) - sum(x.startswith(split + "/") for x in missing),
                         "questions": n_questions}
    if missing:
        raise ValueError("EventHallusion missing {} annotated videos; examples={}; coverage={}; unmatched_filenames={}".format(
            len(missing), missing[:12], counts, unmatched[:8]))
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate EventHallusion sample IDs")
    fingerprint = []
    for path in sorted(used_files):
        stat = Path(path).stat()
        fingerprint.append((path, stat.st_size, stat.st_mtime_ns))
    contract = {
        "dataset": "EventHallusion", "task": "binary", "expected_records": len(rows),
        "question_files": {split: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                           for split, path in annotations.items()},
        "video_inventory_sha256": hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest(),
        "counts": {"eventhallusion_records": len(rows), "eventhallusion_videos": len(used_files),
                   "missing_videos": 0, "by_split": counts},
    }
    return rows, contract


def validate_eventhallusion_inputs(root, train_data, model, adapter, train_summary, questions_dir=None):
    rows, inputs = load_eventhallusion(root, questions_dir)
    train_rows = read_jsonl(train_data)
    validate_rows(train_rows, "train", 300)
    train_videos = {str(Path(r["video_path"]).resolve()) for r in train_rows}
    if train_videos & {str(Path(r["video_path"]).resolve()) for r in rows}:
        raise ValueError("EventHallusion contains a video path used in training")
    inputs.update(trained_adapter_contract(train_data, model, adapter, train_summary))
    inputs["counts"]["train_video_path_overlap"] = 0
    return rows, inputs


def eventhallusion_metrics(rows, expected_count):
    strict_rows = []
    for row in rows:
        strict = parse_yes_no(row["raw_answer"])
        if row["prediction"] != official_binary_parse(row["raw_answer"]) or row["correct"] != (row["prediction"] == row["gt_answer"]):
            raise ValueError("Inconsistent EventHallusion official scoring")
        if row.get("strict_prediction") != strict:
            raise ValueError("Inconsistent EventHallusion diagnostic scoring")
        strict_rows.append(dict(row, prediction=strict, correct=strict == row["gt_answer"]))
    return {
        "status": "COMPLETED" if len(rows) == expected_count else "PARTIAL",
        "metric": "official_binary_question_accuracy", "overall": question_summary(rows),
        "strict_overall": question_summary(strict_rows),
        "by_subset": {name: question_summary([r for r in rows if r["subset"] == name])
                      for name in sorted({r["subset"] for r in rows})},
        "by_category": {name: question_summary([r for r in rows if r["category"] == name])
                        for name in sorted({r["category"] for r in rows})},
    }


def compare_eventhallusion(base_dir, adapter_dir, output):
    directories = [Path(base_dir), Path(adapter_dir)]
    metadata = [json.loads((d / "run_meta.json").read_text()) for d in directories]
    if [m["variant"] for m in metadata] != ["base", "adapter"]:
        raise ValueError("Expected base and adapter evaluation")
    for key in ("inputs", "protocol", "runtime"):
        if metadata[0][key] != metadata[1][key]:
            raise ValueError("EventHallusion evaluation contracts differ: " + key)
    if metadata[0]["inputs"].get("dataset") != "EventHallusion":
        raise ValueError("Outputs are not EventHallusion")
    expected = metadata[0]["inputs"]["expected_records"]
    base, adapter = [read_jsonl(d / "predictions.jsonl") for d in directories]
    if expected <= 0 or len(base) != expected or len(adapter) != expected or len({r['sample_id'] for r in base}) != expected:
        raise ValueError("Both models must cover every EventHallusion question")
    changes = []
    for before, after in zip(base, adapter):
        for key in ("sample_id", "question", "video_path", "video_id", "subset", "category", "gt_answer", "input_digest"):
            if before[key] != after[key]:
                raise ValueError("EventHallusion prediction inputs differ: " + key)
        if before["correct"] != after["correct"]:
            changes.append({"sample_id": before["sample_id"], "subset": before["subset"],
                            "question": before["question"], "gt_answer": before["gt_answer"],
                            "base_answer": before["raw_answer"], "adapter_answer": after["raw_answer"],
                            "transition": "wrong_to_correct" if after["correct"] else "correct_to_wrong"})
    base_metrics, adapter_metrics = [eventhallusion_metrics(rows, expected) for rows in (base, adapter)]
    report = {"status": "COMPLETED", "dataset": "EventHallusion", "records": expected,
              "protocol": metadata[0]["protocol"], "base": base_metrics, "adapter": adapter_metrics,
              "delta_accuracy_pp": 100 * (adapter_metrics["overall"]["accuracy"] - base_metrics["overall"]["accuracy"]),
              "wrong_to_correct": sum(r["transition"] == "wrong_to_correct" for r in changes),
              "correct_to_wrong": sum(r["transition"] == "correct_to_wrong" for r in changes), "changes": changes}
    atomic_json(output, report)
    return report
