"""Submission plan tests; simulated PBS is not evidence of Gadi job execution."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline_submit", ROOT / "scripts/submit_unified_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PipelineTests(unittest.TestCase):
    def test_real_preflight_only_uses_stdlib_and_does_not_submit(self):
        from test_unified_stdlib import UnifiedContractTests
        with tempfile.TemporaryDirectory() as name:
            project = Path(name)
            artifacts = project / "artifacts"
            artifacts.mkdir()
            UnifiedContractTests().fixture(artifacts)
            model = project / "model"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"qwen3_vl"}')
            (model / "model.safetensors.index.json").write_text('{"weight_map":{"w":"weights.safetensors"}}')
            for filename in ("weights.safetensors", "tokenizer_config.json", "preprocessor_config.json"):
                (model / filename).write_text("fixture")
            (model / "model_revision.json").write_text(json.dumps({"repo_id": "Qwen/Qwen3-VL-8B-Instruct", "revision": "fixture"}))
            event = project / "event"
            for split, count, extras in (("entire", 109, 5), ("mix", 193, 0), ("misleading", 95, 7)):
                directory = event / split
                directory.mkdir(parents=True)
                entries = []
                for number in range(1, count + 1):
                    vid = f"{split}_{number:03d}"
                    (directory / (vid + ".mp4")).touch()
                    questions = [{"question": vid, "answer": "yes"}]
                    if number <= extras:
                        questions.append({"question": vid + " extra", "answer": "no"})
                    entries.append({"id": vid, "length": "00:02", "category": "test", "questions": questions})
                (event / (split + "_questions.json")).write_text(json.dumps(entries))
            archive = project / "env.tar.gz"
            archive.touch()
            env = dict(os.environ, VIDEO_TLDPO_MODEL=str(model), VIDEO_TLDPO_ARCHIVE=str(archive),
                VIDEO_TLDPO_CANONICAL_SPLIT=str(artifacts / "split.json"),
                VIDEO_TLDPO_CANONICAL_EVAL=str(artifacts / "eval_vh.jsonl"),
                VIDEO_TLDPO_CANONICAL_TRAIN=str(artifacts / "train_binary.jsonl"),
                VIDEO_TLDPO_EVENT_ROOT=str(event), VIDEO_TLDPO_EVENT_QUESTIONS=str(event))
            env.pop("VIDEO_TLDPO_OUT", None)
            result = subprocess.run([sys.executable, str(ROOT / "scripts/submit_unified_pipeline.py"),
                "--project", str(project), "--backends", "qwen3", "--preflight-only"],
                env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("records=100", result.stdout)
            self.assertIn("records=409", result.stdout)
            self.assertIn("no jobs submitted", result.stdout)
            self.assertFalse((project / "outputs").exists())
            self.assertFalse((project / "logs").exists())

    def test_complete_graph_has_val_both_tests_and_calibration_after_all_adapters(self):
        plan = pipeline.build_plan(["qwen3", "internvl35"], 2)
        self.assertEqual(len(plan), 17)
        seen = set()
        for node in plan:
            self.assertTrue(set(node["dependencies"]) <= seen)
            seen.add(node["key"])
        final = plan[-1]
        self.assertEqual(len(final["dependencies"]), 6)
        self.assertTrue(all("/adapter/" in key for key in final["dependencies"]))
        for node in plan:
            if node.get("command") == "evaluate" and node["variant"] == "adapter":
                self.assertIn(node["backend"] + "/train/2", node["dependencies"])
                self.assertIn(node["backend"] + "/base/" + node["dataset"] + "/" + node["split"], node["dependencies"])

    def test_submit_forwards_real_afterok_graph_and_refuses_duplicate_pipeline(self):
        with tempfile.TemporaryDirectory() as name:
            project = Path(name)
            args = SimpleNamespace(project=project, run_root=project / "outputs/new_run", resume=False,
                                   backends=["qwen3"], walltime="00:30:00", seconds=1800)
            commands = []
            def qsub(command, env, capture=False, timeout=120):
                commands.append((command, env))
                return f"{100 + len(commands)}.gadi-pbs"
            with patch.object(pipeline, "call", side_effect=qsub):
                pipeline.submit(args, {"USER": "fixture"}, pipeline.build_plan(args.backends, 1))
                with self.assertRaisesRegex(ValueError, "already submitted"):
                    pipeline.submit(args, {"USER": "fixture"}, pipeline.build_plan(args.backends, 1))
            self.assertEqual(len(commands), 8)
            self.assertEqual(commands[0][1]["VIDEO_TLDPO_COMMAND"], "train")
            adapter = next(command for command, env in commands if env.get("VIDEO_TLDPO_EVAL_VARIANT") == "adapter")
            self.assertIn("depend=afterok:101.gadi-pbs:102.gadi-pbs", adapter)
            self.assertTrue(commands[-1][0][-1].endswith("calibrate_unified.pbs"))
            self.assertIn("walltime=00:30:00", commands[0][0])

    def test_walltime_validation(self):
        self.assertEqual(pipeline.walltime_seconds("00:30:00"), 1800)
        for value in ("00:00:10", "00:99:00", "bad"):
            with self.assertRaises(Exception):
                pipeline.walltime_seconds(value)


if __name__ == "__main__":
    unittest.main()
