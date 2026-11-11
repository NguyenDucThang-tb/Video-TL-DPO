"""Audit regression cases use tiny local files; no model, torch or GPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("artifact_audit", ROOT / "scripts/audit_unified_run.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture(project):
    data = {}
    for split, count in (("train", 150), ("val", 50), ("test", 155)):
        data[split] = []
        for number in range(count):
            pair = f"temporal/{split}_{number}"
            video = project / "videos" / f"{split}_{number}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_text(str(video))
            for role, label in (("basic", "yes"), ("hallucination", "no")):
                data[split].append(dict(sample_id=pair + "|" + role, pair_id=pair,
                    question=pair + role, gt_answer=label, split=split, video_path=str(video)))
    for split, name in (("train", "train_binary"), ("val", "val_binary"), ("test", "test_records")):
        write_rows(project / "artifacts" / (name + ".jsonl"), data[split])
    hashes = {split: audit_module.sha256(project / "artifacts" / (name + ".jsonl"))
              for split, name in (("train", "train_binary"), ("val", "val_binary"), ("test", "test_records"))}
    event = []
    for number in range(409):
        video = project / "videos" / f"event_{number % 397}.mp4"
        video.write_text(str(video))
        event.append(dict(sample_id=f"eh/entire/entire_{number}|0", pair_id=f"eh/{number}",
            question=f"event question {number}", gt_answer="yes" if number % 2 else "no", video_path=str(video)))
    for backend in ("qwen3", "internvl35"):
        root = project / "outputs/unified_adaptive_seed_v2" / backend
        adapter = root / "train/trainer_state/latest/adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_text(backend)
        write_json(adapter / "adapter_config.json", {"backend": backend})
        contract = {"protocol": "unified_adaptive_seed_v2", "backend": backend,
                    "vh_data": hashes, "canonical_train_sha256": hashes["train"]}
        write_json(root / "train/train_summary.json", {"status": "COMPLETED", "records": 300,
            "optimizer_steps": 38, "adapter_checkpoint": str(adapter), "contract": contract})
        write_json(adapter.parent / "state.json", {"contract": contract, "next_record_index": 300,
            "optimizer_steps": 38, "losses": [0.69] * 300, "order": [r["sample_id"] for r in data["train"]]})
        for dataset, rows in (("videohallucer", data["test"]), ("eventhallusion", event)):
            out = root / (dataset + "_test")
            for variant in ("base", "adapter"):
                meta = dict(contract, variant=variant, dataset=dataset, split="test")
                if variant == "adapter":
                    meta.update(adapter_sha256=audit_module.sha256(adapter / "adapter_model.safetensors"),
                                adapter_config_sha256=audit_module.sha256(adapter / "adapter_config.json"))
                predictions = [dict(row, margin=1.0 if row["gt_answer"] == "yes" else -1.0,
                    prediction=row["gt_answer"], correct=True, input_digest=row["sample_id"],
                    frame_indices=list(range(8))) for row in rows]
                metrics = audit_module.independently_score(predictions, len(rows))
                write_rows(out / variant / "margins.jsonl", predictions)
                write_json(out / variant / "run_meta.json", meta)
                write_json(out / variant / "metrics.json", dict(metrics, status="COMPLETED"))
            write_json(out / "comparison.json", {"wrong_to_correct": 0, "correct_to_wrong": 0})


class AuditTests(unittest.TestCase):
    def test_metrics_match_hand_counted_confusion_and_pair_outcomes(self):
        rows = []
        for pair, predictions in enumerate((("yes", "no"), ("yes", "yes"), ("no", "yes"))):
            for role, gt, prediction in zip(("basic", "hallucination"), ("yes", "no"), predictions):
                rows.append(dict(sample_id=f"temporal/{pair}|{role}", pair_id=f"temporal/{pair}",
                    gt_answer=gt, prediction=prediction, correct=prediction == gt,
                    margin=1.0 if prediction == "yes" else -1.0))
        result = audit_module.independently_score(rows, 6)
        self.assertEqual(result["confusion"], {"tp": 2, "fn": 1, "tn": 1, "fp": 2})
        self.assertEqual(result["overall"], {"n": 6, "correct": 3, "accuracy": 0.5, "yes_rate": 4 / 6})
        self.assertEqual(result["complete_pairs"], 3)
        self.assertEqual(result["both_correct_pairs"], 1)
        self.assertEqual(result["neither_correct_pairs"], 1)

    def test_complete_audit_accepts_consistent_saved_artifacts_and_rejects_test_id_in_train_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            self.assertEqual(audit_module.audit(root)["status"], "PASS_FOR_CHECKED_PROPERTIES")
            path = root / "outputs/unified_adaptive_seed_v2/qwen3/train/trainer_state/latest/state.json"
            state = json.loads(path.read_text())
            state["order"][0] = "temporal/test_0|basic"
            write_json(path, state)
            result = audit_module.audit(root)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(any("checkpoint order" in error for error in result["errors"]))

    def test_stored_correct_flag_cannot_override_margin_and_label(self):
        row = dict(sample_id="eh/entire/entire_001|0", gt_answer="no",
                   margin=2.0, prediction="yes", correct=True)
        with self.assertRaisesRegex(ValueError, "correctness"):
            audit_module.independently_score([row], 1)

    def test_byte_identical_video_with_different_name_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, copied, unrelated = [root / name for name in ("a.mp4", "renamed.mp4", "other.mp4")]
            first.write_bytes(b"same video")
            copied.write_bytes(first.read_bytes())
            unrelated.write_bytes(b"other file")
            result = audit_module.copied_videos({"train": {str(first)}, "test": {str(copied), str(unrelated)}})
            self.assertEqual(len(result["cross_group_duplicates"]), 1)
            self.assertEqual({entry["path"] for entry in result["cross_group_duplicates"][0]["files"]},
                             {str(first), str(copied)})


if __name__ == "__main__":
    unittest.main()
