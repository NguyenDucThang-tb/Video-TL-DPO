"""Known-answer threshold tests and saved-artifact integration; stdlib only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from video_tldpo.evaluation.calibration import calibrate_backend, select_threshold, shared_id, threshold_metrics
from video_tldpo.evaluation.unified import metrics
from video_tldpo.evaluation.video_validation import file_sha256
from video_tldpo.unified import PROTOCOL_ID, source_fingerprint


def prediction(sid, label, margin, split="val"):
    answer = "yes" if margin > 0 else "no"
    return dict(sample_id=sid, pair_id=sid.rsplit("|", 1)[0], gt_answer=label,
                margin=margin, prediction=answer, correct=answer == label, split=split,
                question=sid, video_path="/video/" + sid.split("|")[0] + ".mp4",
                input_digest=sid, frame_indices=list(range(8)))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture(project):
    root = project / "outputs" / PROTOCOL_ID
    backend = root / "qwen3"
    for variant in ("base", "adapter"):
        for directory, count, split in (("videohallucer_val", 50, "val"),
                                        ("videohallucer_test", 155, "test")):
            rows = []
            for pair in range(count):
                for index, (role, label) in enumerate((("basic", "yes"), ("hallucination", "no"))):
                    if split == "val":
                        margin = 2.0 if role == "basic" else 1.0 if variant == "base" else -1.0
                    else:
                        both, single = (89, 57) if variant == "base" else (113, 37)
                        hit = pair < both or (pair < both + single and index == 0)
                        margin = 2.0 if (label == "yes") == hit else -1.0
                    sid = f"temporal/{split}_{pair}|{role}"
                    rows.append(prediction(sid, label, margin, split))
            folder = backend / directory / variant
            meta = dict(protocol=PROTOCOL_ID, backend="qwen3", variant=variant,
                        dataset="videohallucer", split=split, settings={"seed": 0}, source_sha256=source_fingerprint())
            save_rows(folder / "margins.jsonl", rows)
            save_json(folder / "metrics.json", metrics(rows, 2 * count))
            save_json(folder / "run_meta.json", meta)
            if variant == "base":
                save_rows(project / "artifacts" / ("val_binary.jsonl" if split == "val" else "test_records.jsonl"), rows)
                if split == "test":
                    save_json(folder / "base_check.json", {"status": "PASS"})
        rows = []
        correct = 264 if variant == "base" else 258
        for number in range(409):
            label = "yes" if number % 2 else "no"
            hit = number < correct
            rows.append(prediction(f"eh/entire/entire_{number:03d}|0", label,
                2.0 if (label == "yes") == hit else -1.0, "test"))
        folder = backend / "eventhallusion_test" / variant
        save_rows(folder / "margins.jsonl", rows)
        save_json(folder / "metrics.json", metrics(rows, 409))
        save_json(folder / "run_meta.json", dict(protocol=PROTOCOL_ID, backend="qwen3", variant=variant,
            dataset="eventhallusion", split="test", settings={"seed": 0}, source_sha256=source_fingerprint()))
        if variant == "base":
            save_json(folder / "base_check.json", {"status": "PASS"})
    fingerprints = {"val": file_sha256(project / "artifacts/val_binary.jsonl"),
                    "test": file_sha256(project / "artifacts/test_records.jsonl")}
    for path in backend.glob("*/*/run_meta.json"):
        meta = json.loads(path.read_text())
        meta["vh_data"] = fingerprints
        save_json(path, meta)
    return root


class CalibrationTests(unittest.TestCase):
    def test_threshold_uses_val_midpoints(self):
        rows = [prediction(f"temporal/{i}|basic", label, margin)
                for i, (label, margin) in enumerate((("no", -2.0), ("no", 1.0), ("yes", 2.0), ("yes", 3.0)))]
        result = select_threshold(rows)
        self.assertEqual(result["threshold"], 1.5)
        self.assertEqual(result["val_correct"], 4)
        self.assertEqual(result["candidate_count"], 4)

    def test_ties_choose_nearest_zero_and_stable_sign_if_equal_distance(self):
        rows = [prediction(f"temporal/{i}|basic", label, margin)
                for i, (label, margin) in enumerate((("no", -4.0), ("yes", -2.0), ("no", 1.0), ("yes", 3.0)))]
        self.assertEqual(select_threshold(rows)["threshold"], 2.0)
        for row, margin in zip(rows, (-2.0, -1.0, 1.0, 2.0)):
            row.update(margin=margin, prediction="yes" if margin > 0 else "no")
            row["correct"] = row["prediction"] == row["gt_answer"]
        self.assertEqual(select_threshold(rows)["threshold"], -1.5)

    def test_candidate_set_has_no_exterior_threshold_and_rejects_test_selection(self):
        rows = [prediction(f"temporal/{i}|basic", "no", margin) for i, margin in enumerate((1.0, 2.0))]
        self.assertEqual(select_threshold(rows)["threshold"], 1.5)
        self.assertEqual(select_threshold(rows)["val_correct"], 1)
        rows[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "only VH_VAL"):
            select_threshold(rows)

    def test_strict_greater_than_and_original_margins_unchanged(self):
        rows = [prediction("eh/entire/entire_001|0", "no", 1.5, "test")]
        before = copy.deepcopy(rows)
        self.assertEqual(threshold_metrics(rows, 1.5, 1)["overall"]["correct"], 1)
        self.assertEqual(rows, before)
        self.assertEqual(shared_id("eh/mix/mix_008|0"), "eh/mix_008/0|eh")

    def test_complete_saved_runs_calibrate_both_variants_and_export_without_mutating_margins(self):
        with tempfile.TemporaryDirectory() as name:
            project = Path(name)
            root = fixture(project)
            original = root / "qwen3/videohallucer_test/base/margins.jsonl"
            digest = file_sha256(original)
            report = calibrate_backend(project, root, "qwen3")
            self.assertEqual(report["status"], "COMPLETED")
            self.assertEqual(report["selection"]["base"]["selection_split"], "VH_VAL")
            self.assertEqual(report["selection"]["base"]["threshold"], 1.0)
            self.assertEqual(report["selection"]["adapter"]["threshold"], 0.0)
            self.assertEqual(report["raw"]["base"]["VH_TEST"]["overall"]["correct"], 235)
            self.assertEqual(file_sha256(original), digest)
            self.assertTrue((root / "qwen3/calibration_report.json").is_file())
            self.assertTrue((root / "qwen3/calibration/base/calibrated.csv").is_file())
            export = (root / "qwen3/calibration/base/margins.jsonl").read_text().splitlines()
            self.assertEqual(len(export), 819)

    def test_cross_run_pixel_digest_and_source_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            project = Path(name)
            root = fixture(project)
            path = root / "qwen3/videohallucer_val/adapter/margins.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["input_digest"] = "different video input"
            save_rows(path, rows)
            with self.assertRaisesRegex(ValueError, "inputs differ"):
                calibrate_backend(project, root, "qwen3")

    def test_changing_test_predictions_does_not_change_validation_thresholds(self):
        with tempfile.TemporaryDirectory() as name:
            project = Path(name)
            root = fixture(project)
            before = calibrate_backend(project, root, "qwen3")["selection"]
            for variant in ("base", "adapter"):
                folder = root / "qwen3/eventhallusion_test" / variant
                path = folder / "margins.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                # Flip one correct and one wrong answer, preserving required base counts.
                for row in (rows[0], rows[-1]):
                    row["margin"] *= -1
                    row["prediction"] = "yes" if row["margin"] > 0 else "no"
                    row["correct"] = row["prediction"] == row["gt_answer"]
                save_rows(path, rows)
                save_json(folder / "metrics.json", metrics(rows, 409))
            after = calibrate_backend(project, root, "qwen3")["selection"]
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
