"""Contract regressions runnable with system Python; no model/GPU mocks passed as E2E."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import os
import subprocess
import sys

from video_tldpo.data.unified import load_vh, sample_id
from video_tldpo.evaluation.unified import base_check, metrics, restore_margins
from video_tldpo.unified import ANSWER_WORDS, SETTINGS, frame_indices, lora_target_names, optimizer_budget


class UnifiedContractTests(unittest.TestCase):
    def test_short_clips_repeat_to_eight_instead_of_changing_video_budget(self):
        self.assertEqual(frame_indices(1), [0] * 8)
        self.assertEqual(frame_indices(6), [0, 1, 1, 2, 3, 4, 4, 5])
        self.assertEqual(frame_indices(8), list(range(8)))
        self.assertEqual(frame_indices(100)[-1], 99)
        with self.assertRaises(ValueError):
            frame_indices(0)

    def test_budget_and_shared_yes_no_case(self):
        self.assertEqual(optimizer_budget(300), (38, 1))
        self.assertEqual(ANSWER_WORDS["qwen3"], ("Yes", "No"))
        self.assertEqual(ANSWER_WORDS["qwen25"], ("Yes", "No"))
        self.assertEqual(SETTINGS["weight_decay"], 0)

    def test_attention_and_projector_selected_but_language_mlp_excluded(self):
        names = ["model.language_model.layers.0.self_attn.q_proj",
                 "model.language_model.layers.0.mlp.down_proj",
                 "model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2",
                 "model.visual.merger.norm", "visual.blocks.0.attn.qkv"]
        self.assertEqual(lora_target_names(names, "qwen3"), [names[0], names[2], names[3]])
        iv = ["language_model.model.layers.0.self_attn.o_proj", "mlp1.1", "mlp1.3", "mlp1.0"]
        self.assertEqual(lora_target_names(iv, "internvl35"), iv[:3])

    def test_eventhallusion_ids_align_with_adaptive_and_opa_formats(self):
        self.assertEqual(sample_id({"id": "eh/entire_038/1|eh"}), "eh/entire/entire_038|1")
        self.assertEqual(sample_id({"sample_id": "eh/mix/mix_008|0"}), "eh/mix/mix_008|0")

    def fixture(self, root):
        ids, splits = {}, {}
        for split, count in (("train", 150), ("val", 50), ("test", 155)):
            rows = []
            ids[split] = []
            for i in range(count):
                subset = "temporal" if i % 2 else "object_relation"
                pid = "{}/{}_{}".format(subset, split, i)
                ids[split].append(pid)
                video = root / subset / "videos" / (split + "_" + str(i) + ".mp4")
                video.parent.mkdir(parents=True, exist_ok=True)
                video.touch()
                for role, answer in (("basic", "yes"), ("hallucination", "no")):
                    rows.append(dict(sample_id=pid + "|" + role, pair_id=pid, subset=subset,
                        split=split, question="Question " + pid + role, gt_answer=answer,
                        video_path=str(video)))
            splits[split] = rows
            name = {"train": "train_binary.jsonl", "val": "val_binary.jsonl", "test": "test_records.jsonl"}[split]
            (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        (root / "split.json").write_text(json.dumps(ids))
        (root / "eval_vh.jsonl").write_text("".join(json.dumps(r) + "\n" for r in splits["val"] + splits["test"]))
            (root / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in splits["train"]))
        return ids, splits

    def test_canonical_membership_not_just_containment_in_all_predictions(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            ids, splits = self.fixture(root)
            actual, _ = load_vh(root, root / "split.json", root / "eval_vh.jsonl", check_files=False)
            self.assertEqual(actual["test"], splits["test"])
            # Swapping a train/test pair preserves counts AND containment in all eval rows.
            ids["train"][0], ids["test"][0] = ids["test"][0], ids["train"][0]
            (root / "split.json").write_text(json.dumps(ids))
            with self.assertRaisesRegex(ValueError, "membership"):
                load_vh(root, root / "split.json", root / "eval_vh.jsonl", check_files=False)

    def test_matching_ids_with_different_questions_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.fixture(root)
            path = root / "eval_vh.jsonl"
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            rows[-1]["question"] += " changed"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "metadata"):
                load_vh(root, root / "split.json", path, check_files=False)

    def test_base_gate_uses_configured_counts_and_keeps_margin_diagnostics(self):
        rows = []
        for i in range(155):
            # Qwen3 target: 89 both-correct, 57 single-correct, 9 neither-correct.
            for j, answer in enumerate(("yes", "no")):
                correct = i < 89 or (i < 146 and j == 0)
                pred = answer if correct else ("no" if answer == "yes" else "yes")
                rows.append(dict(sample_id="temporal/{}|{}".format(i, "basic" if j == 0 else "hallucination"),
                    pair_id="temporal/{}".format(i), question="q", gt_answer=answer,
                    prediction=pred, correct=correct, margin=1.0 if pred == "yes" else -1.0))
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "margins.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(base_check(rows, metrics(rows, 310), "qwen3", "videohallucer", path, {})["status"], "PASS")
            changed = copy.deepcopy(rows)
            for r in changed[192:194]:
                r["margin"] *= -1
                r["prediction"] = "yes" if r["margin"] > 0 else "no"
                r["correct"] = not r["correct"]
            self.assertEqual(metrics(changed, 310)["overall"]["correct"], 235)
            self.assertEqual(metrics(changed, 310)["both_correct_pairs"], 89)
            check = base_check(changed, metrics(changed, 310), "qwen3", "videohallucer", path, {})
            self.assertEqual(check["status"], "PASS")
            self.assertEqual(len(check["prediction_disagreements"]), 2)
            changed[192]["margin"] *= -1
            changed[192]["prediction"] = "yes" if changed[192]["margin"] > 0 else "no"
            changed[192]["correct"] = not changed[192]["correct"]
            self.assertEqual(base_check(changed, metrics(changed, 310), "qwen3", "videohallucer",
                                        path, {})["status"], "FAIL")

    def test_submission_is_a_real_one_hour_resume_chain_and_preflight_does_not_submit(self):
        script = Path(__file__).resolve().parents[1] / "scripts/submit_unified.sh"
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bins = root / "bin"
            bins.mkdir()
            for command, body in {
                "python-probe": "exit 0\n",
                "qselect": "exit 0\n",
                "qsub": 'echo "$*" >> "$MOCK_CALLS"\necho "123.gadi-pbs"\n',
            }.items():
                path = bins / command
                path.write_text("#!/bin/bash\n" + body)
                path.chmod(0o755)
            archive = root / "env.tar.gz"
            archive.touch()
            calls = root / "calls"
            env = dict(os.environ, PATH=str(bins) + ":" + os.environ["PATH"],
                VIDEO_TLDPO_PROJECT=str(root), VIDEO_TLDPO_PYTHON=str(bins / "python-probe"),
                VIDEO_TLDPO_ARCHIVE=str(archive), MOCK_CALLS=str(calls), USER="fixture")
            args = ["bash", str(script), "train", "qwen3", "--jobs", "2"]
            result = subprocess.run(args + ["--preflight-only"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(calls.exists())
            result = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            submitted = calls.read_text().splitlines()
            self.assertEqual(len(submitted), 2)
            self.assertNotIn("depend=", submitted[0])
            self.assertIn("depend=afterok:123.gadi-pbs", submitted[1])
            self.assertTrue(all("run_unified.pbs" in line and "-v VIDEO_TLDPO_PROJECT" in line for line in submitted))

    def test_actual_preflight_needs_no_ml_libraries_and_training_requires_verified_base(self):
        script = Path(__file__).resolve().parents[1] / "scripts/run_unified.py"
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, splits = self.fixture(root)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"qwen2_5_vl"}')
            (model / "model.safetensors.index.json").write_text('{"weight_map":{"w":"weights.safetensors"}}')
            for filename in ("weights.safetensors", "tokenizer_config.json", "preprocessor_config.json"):
                (model / filename).write_text("fixture")
            ref = root / "reference.jsonl"
            ref.write_text("".join(json.dumps(dict(r, margin=1)) + "\n" for r in splits["test"]))
            args = [sys.executable, str(script), "evaluate", "--backend", "qwen25", "--model", str(model),
                    "--project", str(root), "--artifacts", str(root), "--canonical-split", str(root / "split.json"),
                    "--canonical-eval", str(root / "eval_vh.jsonl"), "--canonical-train", str(root / "sft.jsonl"),
                    "--reference-margins", str(ref), "--preflight-only"]
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("UNIFIED_PREFLIGHT: PASS", result.stdout)
            args[2] = "train"
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("UNIFIED_PREFLIGHT: PASS", result.stdout)
            self.assertFalse((root / "outputs").exists())

    def test_eval_resume_discards_uncommitted_final_record_and_rejects_changed_inputs(self):
        row = dict(sample_id="temporal/1|basic", pair_id="temporal/1", question="q",
                   video_path="v.mp4", gt_answer="yes", margin=1.0, prediction="yes", correct=True)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "margins.jsonl"
            path.write_text(json.dumps(row))
            self.assertEqual(restore_margins(path, [row]), [])
            self.assertEqual(path.read_bytes(), b"")
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(restore_margins(path, [row]), [row])
            with self.assertRaisesRegex(ValueError, "canonical samples"):
                restore_margins(path, [dict(row, question="changed")])


if __name__ == "__main__":
    unittest.main()
