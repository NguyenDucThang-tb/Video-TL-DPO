"""Fit one Yes/No threshold on VH_VAL only; apply it unchanged to both tests."""
import csv
import json
import math
from pathlib import Path

from video_tldpo.data.unified import normalize
from video_tldpo.evaluation.unified import metrics
from video_tldpo.evaluation.video_validation import atomic_json, file_sha256, locked_output, read_jsonl
from video_tldpo.unified import BASE_COUNTS, CALIBRATION_ID, PROTOCOL_ID, source_fingerprint


SETS = (("VH_VAL", "videohallucer_val", 100),
        ("VH_TEST", "videohallucer_test", 310),
        ("EH_ALL", "eventhallusion_test", 409))
COMMON = ("protocol", "settings", "backend", "model_identity", "vh_data", "runtime", "source_sha256")


def select_threshold(validation):
    if not validation or any(row.get("split") != "val" or row["sample_id"].startswith("eh/")
                             for row in validation):
        raise ValueError("Threshold selection accepts only VH_VAL rows")
    metrics(validation, len(validation))
    values = sorted(float(row["margin"]) for row in validation)
    # Include repeated adjacent values as specified; do not add exterior cutoffs.
    candidates = {0.0} | {left / 2 + right / 2 for left, right in zip(values, values[1:])}
    ordered = sorted(candidates, key=lambda threshold: (abs(threshold), threshold))
    counts = {threshold: sum(("yes" if row["margin"] > threshold else "no") == row["gt_answer"]
                            for row in validation) for threshold in ordered}
    best = max(ordered, key=lambda threshold: counts[threshold])
    return {"threshold": best, "val_correct": counts[best], "val_n": len(validation),
            "val_accuracy": counts[best] / len(validation), "candidate_count": len(candidates),
            "selection_split": "VH_VAL", "rule": CALIBRATION_ID,
            "tie_break": "nearest zero; negative first only for equal absolute distance"}


def threshold_metrics(rows, threshold, expected):
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    adjusted = []
    for row in rows:
        prediction = "yes" if row["margin"] > threshold else "no"
        # Work on copies; the original margins and their files stay intact.
        adjusted.append(dict(row, margin=row["margin"] - threshold, prediction=prediction,
                             correct=prediction == row["gt_answer"]))
    return metrics(adjusted, expected)


def shared_id(sample_id):
    if sample_id.startswith("eh/"):
        prefix, question = sample_id.rsplit("|", 1)
        return "eh/{}/{}|eh".format(prefix.rsplit("/", 1)[-1], question)
    return sample_id


def _csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _summary(backend, variant, mode, threshold, scores):
    return {"backend": backend, "variant": variant, "mode": mode, "threshold": threshold,
            "VH_Q_acc": scores["VH_TEST"]["overall"]["accuracy"],
            "VH_pair_acc": scores["VH_TEST"]["pair_accuracy"],
            "EH_acc": scores["EH_ALL"]["overall"]["accuracy"],
            "VH_yes_rate": scores["VH_TEST"]["overall"]["yes_rate"],
            "EH_yes_rate": scores["EH_ALL"]["overall"]["yes_rate"]}


SUMMARY_FIELDS = ("backend", "variant", "mode", "threshold", "VH_Q_acc", "VH_pair_acc",
                  "EH_acc", "VH_yes_rate", "EH_yes_rate")


def calibrate_backend(project, run_root, backend):
    with locked_output(Path(run_root) / backend / "calibration"):
        return _calibrate_backend(project, run_root, backend)


def _calibrate_backend(project, run_root, backend):
    root = Path(run_root) / backend
    rows, metadata, raw, provenance = {}, {}, {}, {}
    val_canonical = {row["sample_id"]: row for row in map(normalize, read_jsonl(Path(project) / "artifacts/val_binary.jsonl"))}
    test_canonical = {row["sample_id"]: row for row in map(normalize, read_jsonl(Path(project) / "artifacts/test_records.jsonl"))}
    artifact_hashes = {"val": file_sha256(Path(project) / "artifacts/val_binary.jsonl"),
                       "test": file_sha256(Path(project) / "artifacts/test_records.jsonl")}
    current_source = source_fingerprint()
    for variant in ("base", "adapter"):
        rows[variant], metadata[variant], raw[variant], provenance[variant] = {}, {}, {}, {}
        for name, directory, expected in SETS:
            folder = root / directory / variant
            pred = read_jsonl(folder / "margins.jsonl")
            meta = json.loads((folder / "run_meta.json").read_text())
            result = metrics(pred, expected)
            if result["status"] != "COMPLETED":
                raise ValueError("Calibration requires complete margins: " + str(folder))
            saved = json.loads((folder / "metrics.json").read_text())
            if saved != result:
                raise ValueError("Saved metrics differ from margins: " + str(folder))
            dataset, split = ("eventhallusion", "test") if name == "EH_ALL" else ("videohallucer", "val" if name == "VH_VAL" else "test")
            if (meta.get("protocol"), meta.get("backend"), meta.get("variant"), meta.get("dataset"), meta.get("split")) != (PROTOCOL_ID, backend, variant, dataset, split):
                raise ValueError("Unexpected calibration source metadata: " + str(folder))
            if any(meta.get("vh_data", {}).get(key) != digest for key, digest in artifact_hashes.items()):
                raise ValueError("VH artifacts changed since evaluation")
            if meta.get("source_sha256") != current_source:
                raise ValueError("Runtime source changed since evaluation; use a consistent run")
            if name != "EH_ALL":
                canonical = val_canonical if name == "VH_VAL" else test_canonical
                if {row["sample_id"] for row in pred} != set(canonical) or len(canonical) != expected:
                    raise ValueError("Calibration rows differ from authoritative VH split")
                for row in pred:
                    source = canonical[row["sample_id"]]
                    if any(row[key] != source[key] for key in ("question", "gt_answer", "pair_id")) or Path(row["video_path"]).resolve() != Path(source["video_path"]).resolve():
                        raise ValueError("Calibration sample content differs from VH artifacts")
            if variant == "base" and name != "VH_VAL":
                check = json.loads((folder / "base_check.json").read_text())
                counts = BASE_COUNTS[backend]
                actual = result["overall"]["correct"] == (counts[2] if name == "EH_ALL" else counts[0])
                if name == "VH_TEST":
                    actual = actual and result["both_correct_pairs"] == counts[1]
                if check.get("status") != "PASS" or not actual:
                    raise ValueError("Base reproduction must pass before reporting calibrated comparisons")
            rows[variant][name], metadata[variant][name], raw[variant][name] = pred, meta, result
            provenance[variant][name] = {"margins_sha256": file_sha256(folder / "margins.jsonl"),
                "run_meta_sha256": file_sha256(folder / "run_meta.json")}
        reference = metadata[variant]["VH_VAL"]
        for name, meta in metadata[variant].items():
            keys = COMMON + (("adapter_sha256", "adapter_config_sha256") if variant == "adapter" else ())
            if any(meta.get(key) != reference.get(key) for key in keys):
                raise ValueError("Model/source/runtime changed between validation and test: " + name)
    for name, _, _ in SETS:
        before, after = rows["base"][name], rows["adapter"][name]
        if any(metadata["base"][name].get(key) != metadata["adapter"][name].get(key) for key in COMMON):
            raise ValueError("Base/adapter contracts differ: " + name)
        if metadata["base"][name].get("event_data") != metadata["adapter"][name].get("event_data"):
            raise ValueError("Base/adapter EventHallusion inventory differs")
        for left, right in zip(before, after):
            if any(left[key] != right[key] for key in ("sample_id", "pair_id", "question", "gt_answer", "input_digest", "frame_indices")):
                raise ValueError("Base/adapter inputs differ: " + name)
    selection, calibrated, summary = {}, {}, []
    csv_rows = []
    for variant in ("base", "adapter"):
        selection[variant] = select_threshold(rows[variant]["VH_VAL"])
        threshold = selection[variant]["threshold"]
        selection[variant]["validation_margins_sha256"] = provenance[variant]["VH_VAL"]["margins_sha256"]
        calibrated[variant] = {name: threshold_metrics(rows[variant][name], threshold, expected)
                               for name, _, expected in SETS}
        folder = root / "calibration" / variant
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / "threshold.json", selection[variant])
        variant_csv = []
        export = []
        for name, _, _ in SETS:
            score = calibrated[variant][name]
            variant_csv.append({"variant": variant, "split": name, "n": score["overall"]["n"],
                "q_acc": score["overall"]["accuracy"], "pair_acc": score.get("pair_accuracy"),
                "yes_rate": score["overall"]["yes_rate"], "threshold": threshold})
            export.extend({"id": shared_id(row["sample_id"]), "sample_id": row["sample_id"],
                "margin": row["margin"], "answer": row["gt_answer"], "split": name}
                          for row in rows[variant][name])
        _csv(folder / "calibrated.csv", variant_csv, ("variant", "split", "n", "q_acc", "pair_acc", "yes_rate", "threshold"))
        temporary = folder / "margins.jsonl.tmp"
        temporary.write_text("".join(json.dumps(row) + "\n" for row in export))
        temporary.replace(folder / "margins.jsonl")
        csv_rows.extend(variant_csv)
        summary.append(_summary(backend, variant, "raw", 0.0, raw[variant]))
        summary.append(_summary(backend, variant, "calibrated", threshold, calibrated[variant]))
    deltas = {}
    for label, left, right in (("base_calibrated_minus_base", summary[1], summary[0]),
                              ("adapter_raw_minus_base_calibrated", summary[2], summary[1]),
                              ("adapter_calibrated_minus_base_calibrated", summary[3], summary[1])):
        deltas[label] = {key + "_pp": 100 * (left[key] - right[key]) for key in ("VH_Q_acc", "VH_pair_acc", "EH_acc")}
    report = {"status": "COMPLETED", "protocol": PROTOCOL_ID, "calibration_rule": CALIBRATION_ID,
              "backend": backend, "selection": selection, "raw": raw, "calibrated": calibrated,
              "summary": summary, "deltas": deltas, "source_files": provenance}
    atomic_json(root / "calibration_report.json", report)
    _csv(root / "calibration_summary.csv", csv_rows, ("variant", "split", "n", "q_acc", "pair_acc", "yes_rate", "threshold"))
    return report


def write_summary(path, reports):
    _csv(Path(path), [row for report in reports for row in report["summary"]], SUMMARY_FIELDS)
