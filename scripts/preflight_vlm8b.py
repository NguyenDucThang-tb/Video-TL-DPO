"""Python 3.6-compatible, stdlib-only Gadi preflight for a new model arm."""
from __future__ import print_function
import json
import os
import sys


def read_jsonl(path):
    with open(path, "r") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def check_rows(path, split, count):
    rows = read_jsonl(path)
    assert len(rows) == count, (path, len(rows), count)
    assert len(set(x["sample_id"] for x in rows)) == count
    assert all(x.get("split") == split for x in rows)
    assert all(os.path.isfile(x["video_path"]) for x in rows)
    return rows


def main():
    backend, model, artifacts = sys.argv[1:]
    config = json.load(open(os.path.join(model, "config.json")))
    expected_type = {"qwen3": "qwen3_vl", "internvl35": "internvl_chat"}[backend]
    assert config.get("model_type") == expected_type, config.get("model_type")
    revision = json.load(open(os.path.join(model, "model_revision.json")))
    expected_repo = {"qwen3": "Qwen/Qwen3-VL-8B-Instruct",
                     "internvl35": "OpenGVLab/InternVL3_5-8B-Instruct"}[backend]
    assert revision.get("repo_id") == expected_repo and revision.get("revision")
    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))
    shards = set(index["weight_map"].values())
    assert shards and all(os.path.isfile(os.path.join(model, name)) for name in shards)
    if backend == "internvl35":
        for name in ("modeling_internvl_chat.py", "modeling_intern_vit.py",
                     "configuration_internvl_chat.py", "configuration_intern_vit.py"):
            assert os.path.isfile(os.path.join(model, name)), name
    train = check_rows(os.path.join(artifacts, "train_binary.jsonl"), "train", 300)
    val = check_rows(os.path.join(artifacts, "val_binary.jsonl"), "val", 100)
    test = check_rows(os.path.join(artifacts, "test_records.jsonl"), "test", 310)
    manifest = json.load(open(os.path.join(artifacts, "split_manifest.json")))
    for rows, split in ((train, "train"), (val, "val"), (test, "test")):
        assert set(x["pair_id"] for x in rows) == set(manifest["pair_ids"][split])
    videos = [{os.path.realpath(x["video_path"]) for x in rows} for rows in (train, val, test)]
    assert not (videos[0] & videos[1] or videos[0] & videos[2] or videos[1] & videos[2])
    print("model_backend:", backend, "model_type:", expected_type, "weight_shards:", len(shards))
    print("records: train=300 val=100 test=310")
    print("video_leakage: 0")
    print("VLM8B_PREFLIGHT: PASS")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("qwen3", "internvl35"):
        raise SystemExit("usage: preflight_vlm8b.py {qwen3|internvl35} MODEL_DIR ARTIFACTS_DIR")
    main()
