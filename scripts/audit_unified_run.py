"""Independent, stdlib-only audit of saved unified runs; never loads a model.

Checks the actual checkpoint order, dataset fingerprints, saved predictions,
input equality and independently recomputed metrics. Optional video hashing
detects byte-identical copies with different paths/names (not re-encodings).
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_rows(rows):
    indexed = {row["sample_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate sample IDs")
    return indexed


def independently_score(rows, expected):
    """Recompute from labels and margin sign, never trust stored correct flags."""
    index_rows(rows)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} predictions, found {len(rows)}")
    confusion = Counter()
    roles = defaultdict(list)
    pairs = defaultdict(dict)
    yes = correct = 0
    for row in rows:
        margin = float(row["margin"])
        if not math.isfinite(margin) or row["gt_answer"] not in ("yes", "no"):
            raise ValueError("Invalid margin/label: " + row["sample_id"])
        prediction = "yes" if margin > 0 else "no"
        hit = prediction == row["gt_answer"]
        if row.get("prediction") != prediction or row.get("correct") != hit:
            raise ValueError("Stored prediction/correctness differs: " + row["sample_id"])
        correct += hit
        yes += prediction == "yes"
        confusion[(row["gt_answer"], prediction)] += 1
        if not row["sample_id"].startswith("eh/"):
            role = row["sample_id"].rsplit("|", 1)[-1]
            if role not in ("basic", "hallucination"):
                raise ValueError("Unknown VH role")
            roles[role].append(hit)
            pair = row["pair_id"]
            if role in pairs[pair]:
                raise ValueError("Duplicate role within a pair")
            pairs[pair][role] = hit
    result = {"overall": {"n": len(rows), "correct": correct,
              "accuracy": correct / len(rows), "yes_rate": yes / len(rows)},
              "confusion": {"tp": confusion[("yes", "yes")],
                            "fn": confusion[("yes", "no")],
                            "tn": confusion[("no", "no")],
                            "fp": confusion[("no", "yes")]}}
    if pairs:
        if any(set(pair) != {"basic", "hallucination"} for pair in pairs.values()):
            raise ValueError("Incomplete VH pair")
        both = sum(all(pair.values()) for pair in pairs.values())
        neither = sum(not any(pair.values()) for pair in pairs.values())
        result.update(complete_pairs=len(pairs), both_correct_pairs=both,
                      pair_accuracy=both / len(pairs), neither_correct_pairs=neither,
                      by_role={role: {"n": len(hits), "correct": sum(hits),
                                     "accuracy": sum(hits) / len(hits)}
                               for role, hits in roles.items()})
    return result


def copied_videos(groups):
    """Hash only file sizes present in more than one split/dataset."""
    sizes = defaultdict(list)
    for group, paths in groups.items():
        for path in sorted(paths):
            sizes[Path(path).stat().st_size].append((group, path))
    matches = []
    hashed = total_bytes = 0
    for size, candidates in sizes.items():
        if len({group for group, _ in candidates}) < 2:
            continue
        digests = defaultdict(list)
        for group, path in candidates:
            digests[sha256(path)].append({"group": group, "path": path})
            hashed += 1
            total_bytes += size
        for digest, entries in digests.items():
            if len({entry["group"] for entry in entries}) > 1:
                matches.append({"sha256": digest, "files": entries})
    return {"status": "RUN", "files_hashed": hashed,
            "bytes_hashed": total_bytes, "cross_group_duplicates": matches}


def audit(project, hash_videos=False):
    errors = []
    report = {"project": str(project), "errors": errors}

    def require(condition, message):
        if not condition:
            errors.append(message)

    files = {"train": "train_binary.jsonl", "val": "val_binary.jsonl",
             "test": "test_records.jsonl"}
    data = {name: read_jsonl(project / "artifacts" / filename)
            for name, filename in files.items()}
    indices = {name: index_rows(rows) for name, rows in data.items()}
    hashes = {name: sha256(project / "artifacts" / filename)
              for name, filename in files.items()}
    groups = {name: {str(Path(row["video_path"]).resolve()) for row in rows}
              for name, rows in data.items()}
    for name, paths in groups.items():
        require(all(Path(path).is_file() for path in paths), f"{name}: inaccessible video file/mount")
    for name, count in (("train", 300), ("val", 100), ("test", 310)):
        require(len(data[name]) == count, f"{name}: wrong record count")
        require(all(row.get("split") == name for row in data[name]), f"{name}: wrong split label")
        for row in data[name]:
            role = row["sample_id"].rsplit("|", 1)[-1]
            require(row["sample_id"] == row["pair_id"] + "|" + role,
                    f"{name}: sample/pair ID mismatch")
            require(role in ("basic", "hallucination") and
                    row["gt_answer"] == ("yes" if role == "basic" else "no"),
                    f"{name}: unexpected role/label")
    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        pair_overlap = {r["pair_id"] for r in data[left]} & {r["pair_id"] for r in data[right]}
        overlap[left + "/" + right] = {"sample_ids": len(set(indices[left]) & set(indices[right])),
            "pair_ids": len(pair_overlap), "real_paths": len(groups[left] & groups[right])}
        require(not any(overlap[left + "/" + right].values()), f"Leakage: {left}/{right}")
    report["data"] = {name: {"records": len(rows), "videos": len(groups[name]), "sha256": hashes[name]}
                      for name, rows in data.items()}
    report["overlap"] = overlap
    root = project / "outputs/unified_adaptive_seed_v2"
    report["backends"] = {}
    for backend in ("qwen3", "internvl35"):
        backend_root = root / backend
        summary = json.loads((backend_root / "train/train_summary.json").read_text())
        adapter = Path(summary["adapter_checkpoint"])
        state = json.loads((adapter.parent / "state.json").read_text())
        require(summary["status"] == "COMPLETED" and summary["records"] == 300
                and summary["optimizer_steps"] == 38, f"{backend}: incomplete training")
        require(summary["contract"].get("protocol") == "unified_adaptive_seed_v2",
                f"{backend}: wrong training protocol")
        require(state["contract"] == summary["contract"], f"{backend}: checkpoint/summary contract mismatch")
        require(state["next_record_index"] == 300 and state["optimizer_steps"] == 38
                and len(state["losses"]) == 300, f"{backend}: wrong checkpoint cursor/loss count")
        require(all(math.isfinite(loss) for loss in state["losses"]), f"{backend}: nonfinite saved loss")
        require(len(state["order"]) == 300 and set(state["order"]) == set(indices["train"]),
                f"{backend}: checkpoint order is not exactly the train sample set")
        require(summary["contract"]["canonical_train_sha256"] == hashes["train"],
                f"{backend}: actual train source differs from current artifact")
        for split in files:
            require(summary["contract"]["vh_data"][split] == hashes[split],
                    f"{backend}: {split} artifact changed since training")
        losses = state["losses"]
        result = {"train_order_matches": len(state["order"]) == 300
                  and set(state["order"]) == set(indices["train"]),
                  "train_loss": {"first_sample": losses[0], "last_sample": losses[-1],
                    "mean": sum(losses) / len(losses), "min": min(losses), "max": max(losses),
                    "group_means": [sum(losses[i:i + 8]) / len(losses[i:i + 8])
                                    for i in range(0, len(losses), 8)]}, "datasets": {}}
        report["backends"][backend] = result
        for dataset, count in (("videohallucer", 310), ("eventhallusion", 409)):
            directory = backend_root / (dataset + "_test")
            predictions = {}
            metas = {}
            scores = {}
            for variant in ("base", "adapter"):
                folder = directory / variant
                rows = read_jsonl(folder / "margins.jsonl")
                pred_index = index_rows(rows)
                predictions[variant] = pred_index
                metas[variant] = json.loads((folder / "run_meta.json").read_text())
                meta = metas[variant]
                require(meta["variant"] == variant and meta["backend"] == backend
                        and meta["dataset"] == dataset and meta["split"] == "test",
                        f"{backend}/{dataset}/{variant}: metadata mismatch")
                if variant == "adapter":
                    for key in ("protocol", "settings", "model_identity", "vh_data", "backend", "source_sha256", "canonical_train_sha256"):
                        require(meta.get(key) == summary["contract"].get(key),
                                f"{backend}/{dataset}: evaluation/training contract mismatch: {key}")
                    require(meta["adapter_sha256"] == sha256(adapter / "adapter_model.safetensors")
                            and meta["adapter_config_sha256"] == sha256(adapter / "adapter_config.json"),
                            f"{backend}/{dataset}: evaluated adapter differs from checkpoint")
                score = independently_score(rows, count)
                saved = json.loads((folder / "metrics.json").read_text())
                require(saved["status"] == "COMPLETED", f"{backend}/{dataset}/{variant}: partial metrics")
                for key, value in score["overall"].items():
                    require(math.isclose(saved["overall"][key], value, rel_tol=0, abs_tol=1e-12),
                            f"{backend}/{dataset}/{variant}: incorrect metric {key}")
                for key in ("complete_pairs", "both_correct_pairs", "pair_accuracy"):
                    if key in score:
                        require(math.isclose(saved[key], score[key], rel_tol=0, abs_tol=1e-12),
                                f"{backend}/{dataset}/{variant}: incorrect metric {key}")
                if dataset == "videohallucer":
                    require(set(pred_index) == set(indices["test"]), f"{backend}: test IDs differ")
                    for sid in set(pred_index) & set(indices["test"]):
                        source, pred = indices["test"][sid], pred_index[sid]
                        require(all(source[k] == pred[k] for k in ("question", "gt_answer", "pair_id"))
                                and Path(source["video_path"]).resolve() == Path(pred["video_path"]).resolve(),
                                f"{backend}: canonical test mismatch for {sid}")
                scores[variant] = score
            require(set(predictions["base"]) == set(predictions["adapter"]),
                    f"{backend}/{dataset}: base/adapter IDs differ")
            for key in ("protocol", "settings", "model_identity", "vh_data", "event_data", "runtime", "source_sha256"):
                require(metas["base"].get(key) == metas["adapter"].get(key),
                        f"{backend}/{dataset}: base/adapter contract mismatch: {key}")
            fixed = corrected = regressed = 0
            for sid in set(predictions["base"]) & set(predictions["adapter"]):
                before, after = predictions["base"][sid], predictions["adapter"][sid]
                require(all(before[k] == after[k] for k in ("question", "gt_answer", "video_path", "input_digest", "frame_indices")),
                        f"{backend}/{dataset}: input differs: {sid}")
                corrected += not before["correct"] and after["correct"]
                regressed += before["correct"] and not after["correct"]
                fixed += before["correct"] == after["correct"]
            comparison = json.loads((directory / "comparison.json").read_text())
            require(comparison["wrong_to_correct"] == corrected and comparison["correct_to_wrong"] == regressed,
                    f"{backend}/{dataset}: comparison transition counts differ")
            result["datasets"][dataset] = {"base": scores["base"], "adapter": scores["adapter"],
                "wrong_to_correct": corrected, "correct_to_wrong": regressed,
                "unchanged_correctness": fixed}
            if dataset == "eventhallusion":
                paths = {str(Path(row["video_path"]).resolve()) for row in predictions["base"].values()}
                require(all(Path(path).is_file() for path in paths), "EH: inaccessible video file/mount")
                if "eventhallusion" in groups:
                    require(paths == groups["eventhallusion"], "Backends evaluated different EH videos")
                groups["eventhallusion"] = paths
    eh_overlap = groups["train"] & groups["eventhallusion"]
    report["overlap"]["train/eventhallusion"] = {"real_paths": len(eh_overlap)}
    require(not eh_overlap, "EH video path overlaps training")
    if hash_videos:
        report["video_content"] = copied_videos(groups)
        require(not report["video_content"]["cross_group_duplicates"], "Byte-identical videos across groups")
    else:
        report["video_content"] = {"status": "NOT_RUN", "hint": "Use --hash-videos on a compute/copyq job"}
    report["status"] = "FAIL" if errors else "PASS_FOR_CHECKED_PROPERTIES"
    report["limitations"] = ["Does not detect re-encoded/near-duplicate clips or pretraining contamination.",
        "Saved margin consistency does not independently reproduce model forward logits.",
        "Large gains can reflect answer calibration; this audit does not prove better visual grounding."]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/g/data/hn98") / os.environ.get("USER", "dd9648") / "Video-TL-DPO")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hash-videos", action="store_true")
    args = parser.parse_args()
    output = args.output or args.project / "artifacts/unified_audit.json"
    try:
        report = audit(args.project, args.hash_videos)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"status": "FAIL", "errors": [str(exc)], "scope": "Audit incomplete"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("audit_report:", output)
    return int(report["status"] == "FAIL")


if __name__ == "__main__":
    sys.exit(main())
