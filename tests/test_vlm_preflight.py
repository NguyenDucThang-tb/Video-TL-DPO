import json
from pathlib import Path
import subprocess
import sys


def test_vlm_preflight_accepts_only_fixed_video_disjoint_artifacts(tmp_path):
    project = Path(__file__).resolve().parents[1]
    artifacts = tmp_path / "artifacts"
    model = tmp_path / "model"
    artifacts.mkdir()
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_vl"}')
    (model / "model_revision.json").write_text(json.dumps({
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct", "revision": "abc123"}))
    (model / "tokenizer_config.json").write_text("{}")
    (model / "weights.safetensors").write_bytes(b"weights")
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "weights.safetensors"}}))
    split_ids = {"train": [], "val": [], "test": []}
    split_spec = (("train", 150), ("val", 50), ("test", 155))
    for split, n_pairs in split_spec:
        rows = []
        for i in range(n_pairs):
            pair_id = "{}/{}".format(split, i)
            split_ids[split].append(pair_id)
            video = tmp_path / "{}.mp4".format(pair_id.replace("/", "_"))
            video.touch()
            for role, answer in (("basic", "yes"), ("hallucination", "no")):
                rows.append({"sample_id": pair_id + "|" + role, "pair_id": pair_id,
                             "split": split, "video_path": str(video), "question": "Q?",
                             "gt_answer": answer})
        target = "train_binary.jsonl" if split == "train" else \
            "val_binary.jsonl" if split == "val" else "test_records.jsonl"
        (artifacts / target).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (artifacts / "split_manifest.json").write_text(json.dumps({"pair_ids": split_ids}))
    result = subprocess.run([sys.executable, str(project / "scripts/preflight_vlm8b.py"),
                             "qwen3", str(model), str(artifacts)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "VLM8B_PREFLIGHT: PASS" in result.stdout

    (artifacts / "val_binary.jsonl").write_text(
        (artifacts / "val_binary.jsonl").read_text().replace("val/0", "train/0"))
    result = subprocess.run([sys.executable, str(project / "scripts/preflight_vlm8b.py"),
                             "qwen3", str(model), str(artifacts)],
                            capture_output=True, text=True)
    assert result.returncode != 0
