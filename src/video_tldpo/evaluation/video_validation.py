"""Validation IO and metrics; stdlib-only, including on Gadi's login Python.

Scores describe our fixed, video-disjoint validation split, not the full official
VideoHallucer benchmark. Unparseable answers count as incorrect.
"""
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

PROTOCOL = {
    "version": 1, "frames": "uniform_up_to_8_even", "sampled_fps_forwarded": True,
    "prompt_suffix": " Answer yes or no.", "do_sample": False,
    "num_beams": 1, "max_new_tokens": 16,
    "parser": "leading_yes_no_no_opposite_word_v1",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def role_of(row):
    role = row["sample_id"].rsplit("|", 1)[-1]
    if role not in ("basic", "hallucination"):
        raise ValueError("Unrecognized record role: " + row["sample_id"])
    return role


def validate_rows(rows, split, expected):
    if len(rows) != expected or len({r["sample_id"] for r in rows}) != expected:
        raise ValueError("Expected {} unique {} records".format(expected, split))
    pairs = defaultdict(set)
    for row in rows:
        role = role_of(row)
        if row.get("split") != split or not row.get("question"):
            raise ValueError("Invalid split/question: " + row["sample_id"])
        if row["sample_id"] != row["pair_id"] + "|" + role:
            raise ValueError("sample_id does not match pair_id: " + row["sample_id"])
        if row.get("gt_answer", "").strip().lower() not in ("yes", "no"):
            raise ValueError("Invalid ground truth: " + row["sample_id"])
        if not Path(row["video_path"]).is_file():
            raise ValueError("Missing video: " + row["video_path"])
        pairs[row["pair_id"]].add(role)
    if len(pairs) != expected // 2 or any(v != {"basic", "hallucination"} for v in pairs.values()):
        raise ValueError("Each pair must contain basic and hallucination records")
    return pairs


def validate_inputs(data, train_data, model, adapter, train_summary, manifest=None,
                    split="val", validation_data=None):
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    data, train_data, model, adapter, train_summary = map(
        Path, (data, train_data, model, adapter, train_summary))
    rows, train_rows = read_jsonl(data), read_jsonl(train_data)
    expected = 100 if split == "val" else 310
    eval_pairs = validate_rows(rows, split, expected)
    train_pairs = validate_rows(train_rows, "train", 300)
    manifest = Path(manifest) if manifest is not None else data.parent / "split_manifest.json"
    split_ids = json.loads(manifest.read_text())["pair_ids"]
    if set(split_ids["train"]) != set(train_pairs) or set(split_ids[split]) != set(eval_pairs):
        raise ValueError("Records differ from the published split manifest")
    train_videos = {str(Path(r["video_path"]).resolve()) for r in train_rows}
    eval_videos = {str(Path(r["video_path"]).resolve()) for r in rows}
    if train_videos & eval_videos or set(train_pairs) & set(eval_pairs):
        raise ValueError("Train/evaluation video or pair leakage detected")
    validation_digest = None
    if split == "test":
        validation_data = Path(validation_data) if validation_data is not None else data.parent / "val_binary.jsonl"
        val_rows = read_jsonl(validation_data)
        val_pairs = validate_rows(val_rows, "val", 100)
        val_videos = {str(Path(r["video_path"]).resolve()) for r in val_rows}
        if set(val_pairs) != set(split_ids["val"]):
            raise ValueError("Validation records differ from the published manifest")
        if eval_videos & val_videos or train_videos & val_videos or set(eval_pairs) & set(val_pairs) or set(train_pairs) & set(val_pairs):
            raise ValueError("Train/validation/test video or pair leakage detected")
        validation_digest = file_sha256(validation_data)
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "qwen2_5_vl":
        raise ValueError("Expected Qwen2.5-VL")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if not shards:
        raise ValueError("Model weight index is empty")
    for name in list(shards) + ["tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"]:
        if not (model / name).is_file():
            raise ValueError("Missing model file: " + str(model / name))
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / name).is_file():
            raise ValueError("Missing adapter file: " + str(adapter / name))
    summary = json.loads(train_summary.read_text())
    state = json.loads((adapter.parent / "state.json").read_text())
    if summary.get("status") != "COMPLETED" or summary.get("records") != 300 or state.get("next_record_index") != 300:
        raise ValueError("Adapter is not the completed 300-record checkpoint")
    if Path(summary["adapter_checkpoint"]).resolve() != adapter.resolve():
        raise ValueError("Adapter differs from the completed training summary")
    train_digest = file_sha256(train_data)
    if state.get("contract", {}).get("dataset_sha256") != train_digest:
        raise ValueError("Training JSONL differs from checkpoint provenance")
    if summary.get("contract") != state.get("contract"):
        raise ValueError("Summary/checkpoint contracts differ")
    model_files = {name: file_sha256(model / name) for name in
                   ("config.json", "model.safetensors.index.json", "preprocessor_config.json")}
    if state["contract"].get("model_files") != model_files:
        raise ValueError("Evaluation model differs from the training base model")
    inputs = {
        "validation_sha256": file_sha256(data), "train_sha256": train_digest,
        "manifest_sha256": file_sha256(manifest),
        "model_files": model_files, "adapter_sha256": file_sha256(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json"),
        "counts": {split + "_records": expected, split + "_pairs": len(eval_pairs), split + "_videos": len(eval_videos),
                   "train_records": 300, "train_videos": len(train_videos), "video_overlap": 0},
    }
    if split == "test":
        inputs.update(evaluation_split="test", expected_records=expected,
                      test_sha256=inputs["validation_sha256"], validation_sha256=validation_digest)
        inputs["counts"].update(val_records=100, val_videos=len(val_videos), val_test_video_overlap=0)
    return rows, inputs


def parse_yes_no(text):
    match = re.match(r"^\s*(yes|no)\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    prediction = match.group(1).lower()
    opposite = "no" if prediction == "yes" else "yes"
    if re.search(r"\b" + opposite + r"\b", text[match.end():], flags=re.IGNORECASE):
        return None
    return prediction


@contextmanager
def locked_output(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".eval.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another evaluator is using " + str(directory)) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(str(temporary), str(path))


def initialize_output(directory, contract):
    directory = Path(directory)
    metadata, predictions = directory / "run_meta.json", directory / "predictions.jsonl"
    if metadata.exists():
        if json.loads(metadata.read_text()) != contract:
            raise ValueError("Evaluation contract changed; use a new output directory")
    else:
        if predictions.exists() and predictions.stat().st_size:
            raise ValueError("Existing predictions have no provenance metadata")
        atomic_json(metadata, contract)


def restore_predictions(path, expected_rows):
    path = Path(path)
    if not path.exists():
        return []
    predictions, offset = [], 0
    lines = path.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            prediction = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError("Corrupt predictions JSONL before the incomplete final record")
            with path.open("r+b") as handle:
                handle.truncate(offset)
            print("recovered_partial_prediction_line:", str(path), flush=True)
            break
        if index >= len(expected_rows):
            raise ValueError("Prediction file has too many records")
        row = expected_rows[index]
        for key in ("sample_id", "pair_id", "video_path", "question", "subset"):
            if prediction.get(key) != row[key]:
                raise ValueError("Prediction order/content differs at " + row["sample_id"])
        gt = row["gt_answer"].strip().lower()
        parsed = parse_yes_no(prediction["raw_answer"])
        if prediction.get("gt_answer") != gt or prediction.get("prediction") != parsed \
                or prediction.get("correct") != (parsed == gt) or not prediction.get("input_digest"):
            raise ValueError("Prediction scoring/provenance is inconsistent")
        predictions.append(prediction)
        offset += len(line)
    # A complete final JSON object without a newline must not merge with the next append.
    if lines and len(predictions) == len(lines) and not lines[-1].endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return predictions


def append_prediction(path, prediction):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def question_summary(rows):
    n = len(rows)
    yes = sum(r["prediction"] == "yes" for r in rows)
    invalid = sum(r["prediction"] is None for r in rows)
    correct = sum(bool(r["correct"]) for r in rows)
    confusion = {name: 0 for name in ("tp", "fp", "tn", "fn", "invalid_gt_yes", "invalid_gt_no")}
    for r in rows:
        gt, pred = r["gt_answer"], r["prediction"]
        key = ("invalid_gt_" + gt) if pred is None else \
            ("tp" if gt == "yes" and pred == "yes" else "fn" if gt == "yes" else "tn" if pred == "no" else "fp")
        confusion[key] += 1
    negatives = sum(r["gt_answer"] == "no" for r in rows)
    return {"n": n, "correct": correct, "accuracy": correct / n if n else None,
            "yes_rate": yes / n if n else None, "invalid_answers": invalid,
            "invalid_rate": invalid / n if n else None, "confusion": confusion,
            "false_positive_rate": confusion["fp"] / negatives if negatives else None}


def build_metrics(predictions, expected_count=100):
    groups = {}
    for field in ("subset", "role"):
        groups[field] = {value: question_summary([r for r in predictions if r[field] == value])
                         for value in sorted({r[field] for r in predictions})}
    pairs = defaultdict(list)
    for row in predictions:
        pairs[row["pair_id"]].append(row)
    complete_pairs = [rows for rows in pairs.values() if len(rows) == 2 and {r["role"] for r in rows} == {"basic", "hallucination"}]
    both_correct = sum(all(r["correct"] for r in rows) for rows in complete_pairs)
    return {"status": "COMPLETED" if len(predictions) == expected_count else "PARTIAL",
            "overall": question_summary(predictions), "by_subset": groups["subset"],
            "by_role": groups["role"], "complete_pairs": len(complete_pairs),
            "both_correct_pairs": both_correct,
            "pair_accuracy": both_correct / len(complete_pairs) if complete_pairs else None}


def compare_outputs(base_dir, adapter_dir, output):
    base_dir, adapter_dir = Path(base_dir), Path(adapter_dir)
    metadata = [json.loads((d / "run_meta.json").read_text()) for d in (base_dir, adapter_dir)]
    if [m["variant"] for m in metadata] != ["base", "adapter"]:
        raise ValueError("Expected base and adapter evaluation outputs")
    for key in ("inputs", "protocol", "runtime"):
        if metadata[0][key] != metadata[1][key]:
            raise ValueError("Evaluation protocols/data differ: " + key)
    expected = metadata[0]["inputs"].get("expected_records", 100)
    split = metadata[0]["inputs"].get("evaluation_split", "val")
    if expected != {"val": 100, "test": 310}.get(split):
        raise ValueError("Invalid evaluation split/count in metadata")
    base, adapter = [read_jsonl(d / "predictions.jsonl") for d in (base_dir, adapter_dir)]
    if len(base) != expected or len(adapter) != expected or len({r['sample_id'] for r in base}) != expected:
        raise ValueError("Comparison requires all {} unique predictions per model".format(expected))
    changes = []
    for before, after in zip(base, adapter):
        for key in ("sample_id", "pair_id", "question", "video_path", "subset", "role", "gt_answer", "input_digest"):
            if before[key] != after[key]:
                raise ValueError("Prediction inputs differ: " + key)
        for row in (before, after):
            if row["prediction"] != parse_yes_no(row["raw_answer"]) or row["correct"] != (row["prediction"] == row["gt_answer"]):
                raise ValueError("Inconsistent prediction scoring")
        if before["correct"] != after["correct"]:
            changes.append({"sample_id": before["sample_id"], "subset": before["subset"],
                            "question": before["question"], "gt_answer": before["gt_answer"],
                            "base_answer": before["raw_answer"], "adapter_answer": after["raw_answer"],
                            "transition": "wrong_to_correct" if after["correct"] else "correct_to_wrong"})
    base_metrics, adapter_metrics = build_metrics(base, expected), build_metrics(adapter, expected)
    report = {"status": "COMPLETED", "records": expected, "split": split, "protocol": metadata[0]["protocol"],
              "base": base_metrics, "adapter": adapter_metrics,
              "delta_accuracy_pp": 100 * (adapter_metrics["overall"]["accuracy"] - base_metrics["overall"]["accuracy"]),
              "wrong_to_correct": sum(r["transition"] == "wrong_to_correct" for r in changes),
              "correct_to_wrong": sum(r["transition"] == "correct_to_wrong" for r in changes), "changes": changes}
    atomic_json(output, report)
    return report
