"""Build held-out test question records from existing IDs, never resplit data.

The ordinal ID algorithm matches the Gadi split creation: sorted annotation
files, per-subset deduplication by basic/hallucination video/question signature.
Existing train/val records must match reconstructed IDs before test is written.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from video_tldpo.evaluation.video_validation import (
    file_sha256, locked_output, read_jsonl, validate_rows,
)


def reconstruct_pairs(root):
    root = Path(root)
    pairs = {}
    for subset in ("temporal", "object_relation"):
        directory = root / subset
        if not directory.is_dir():
            raise ValueError("Missing subset directory: " + str(directory))
        seen = {}
        for annotation in sorted(directory.glob("*.json")):
            data = json.loads(annotation.read_text())
            if not isinstance(data, list):
                continue
            for index, item in enumerate(data):
                basic, hallucination = item["basic"], item["hallucination"]
                signature = (basic["video"], basic["question"],
                             hallucination["video"], hallucination["question"])
                answers = tuple(str(q["answer"]).strip().lower() for q in (basic, hallucination))
                if signature in seen:
                    if seen[signature] != answers:
                        raise ValueError("Conflicting duplicate annotations: " + str(annotation))
                    continue
                pair_id = subset + "/" + str(len(seen))
                seen[signature] = answers
                rows = []
                for role, question in (("basic", basic), ("hallucination", hallucination)):
                    video = directory / "videos" / Path(question["video"]).name
                    rows.append({
                        "sample_id": pair_id + "|" + role, "source_dataset": "VideoHallucer",
                        "subset": subset, "pair_id": pair_id, "video_path": str(video),
                        "question": question["question"], "gt_answer": str(question["answer"]).strip().lower(),
                        "provenance": {"annotation_file": str(annotation), "example_index": index, "role": role},
                    })
                if Path(rows[0]["video_path"]).resolve() != Path(rows[1]["video_path"]).resolve():
                    raise ValueError("Basic/hallucination video mismatch: " + pair_id)
                pairs[pair_id] = rows
    return pairs


def build_test(dataset_root, manifest, train_data, val_data, output):
    manifest, output = Path(manifest), Path(output)
    split_ids = json.loads(manifest.read_text())["pair_ids"]
    expected_pairs = {"train": 150, "val": 50, "test": 155}
    for split, expected in expected_pairs.items():
        if len(split_ids[split]) != expected or len(set(split_ids[split])) != expected:
            raise ValueError("Unexpected pair IDs/count for " + split)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(split_ids[a]) & set(split_ids[b]):
            raise ValueError("Pair overlap in manifest: " + a + "/" + b)
    pairs = reconstruct_pairs(dataset_root)
    for pair_id in sum((split_ids[s] for s in ("train", "val", "test")), []):
        if pair_id not in pairs:
            raise ValueError("Manifest pair missing from source annotations: " + pair_id)

    videos = {}
    for split, path, count in (("train", train_data, 300), ("val", val_data, 100)):
        rows = read_jsonl(path)
        actual_pairs = validate_rows(rows, split, count)
        if set(actual_pairs) != set(split_ids[split]):
            raise ValueError("Anchor records differ from manifest: " + split)
        for row in rows:
            source = next(r for r in pairs[row["pair_id"]] if r["sample_id"] == row["sample_id"])
            for key in ("question", "subset", "gt_answer"):
                value = row[key].strip().lower() if key == "gt_answer" else row[key]
                if value != source[key]:
                    raise ValueError("Annotation ID reconstruction differs from anchor: " + row["sample_id"])
            if Path(row["video_path"]).resolve() != Path(source["video_path"]).resolve():
                raise ValueError("Video differs from anchor: " + row["sample_id"])
        videos[split] = {str(Path(row["video_path"]).resolve()) for row in rows}

    rows = []
    manifest_digest = file_sha256(manifest)
    for pair_id in split_ids["test"]:
        for source in pairs[pair_id]:
            row = dict(source, split="test")
            row["provenance"] = dict(source["provenance"], manifest_sha256=manifest_digest)
            rows.append(row)
    validate_rows(rows, "test", 310)
    videos["test"] = {str(Path(row["video_path"]).resolve()) for row in rows}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if videos[a] & videos[b]:
            raise ValueError("Video leakage detected: " + a + "/" + b)

    with locked_output(output.parent):
        if output.exists():
            if read_jsonl(output) != rows:
                raise ValueError("Existing test records differ; refusing to overwrite " + str(output))
            print("existing_test_records: verified unchanged", flush=True)
        else:
            temporary = output.with_name(output.name + ".tmp")
            temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            os.replace(str(temporary), str(output))
    result = {"records": len(rows), "pairs": len(split_ids["test"]), "videos": len(videos["test"]),
              "train_test_video_overlap": 0, "val_test_video_overlap": 0, "output": str(output)}
    print("test_records: PASS", json.dumps(result, sort_keys=True), flush=True)
    return rows


def main():
    project = Path(os.environ.get("VIDEO_TLDPO_PROJECT", "/g/data/hn98/dd9648/Video-TL-DPO"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/scratch/jp09/dd9648/datasets_video_hallu/videohallucer"))
    parser.add_argument("--manifest", type=Path, default=project / "artifacts/split_manifest.json")
    parser.add_argument("--train-data", type=Path, default=project / "artifacts/train_binary.jsonl")
    parser.add_argument("--val-data", type=Path, default=project / "artifacts/val_binary.jsonl")
    parser.add_argument("--output", type=Path, default=project / "artifacts/test_records.jsonl")
    args = parser.parse_args()
    try:
        build_test(args.dataset_root, args.manifest, args.train_data, args.val_data, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, "TEST RECORD BUILD FAILED: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
