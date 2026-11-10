"""Resumable forward-only scoring and canonical base validation."""
import json
import math
from pathlib import Path
import time

from video_tldpo.data.unified import sample_id
from video_tldpo.evaluation.video_validation import atomic_json, file_sha256, locked_output, read_jsonl
from video_tldpo.unified import BASE_COUNTS, PROTOCOL_ID


def metrics(rows, expected):
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate margin sample IDs")
    for r in rows:
        if not math.isfinite(r["margin"]) or r["prediction"] != ("yes" if r["margin"] > 0 else "no"):
            raise ValueError("Invalid margin/prediction")
        if r["correct"] != (r["prediction"] == r["gt_answer"]):
            raise ValueError("Invalid correctness flag")
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    out = {"status": "COMPLETED" if n == expected else "PARTIAL", "protocol": PROTOCOL_ID,
           "overall": {"n": n, "correct": correct, "accuracy": correct / n if n else None,
                       "yes_rate": sum(r["prediction"] == "yes" for r in rows) / n if n else None}}
    if rows and not rows[0]["sample_id"].startswith("eh/"):
        pairs = {}
        for row in rows:
            pairs.setdefault(row["pair_id"], []).append(row)
        complete = [p for p in pairs.values() if len(p) == 2 and
                    {r["sample_id"].rsplit("|", 1)[-1] for r in p} == {"basic", "hallucination"}]
        both = sum(all(r["correct"] for r in p) for p in complete)
        out.update(complete_pairs=len(complete), both_correct_pairs=both,
                   pair_accuracy=both / len(complete) if complete else None)
    return out


def reference_rows(path, rows):
    references = read_jsonl(path)
    by_id = {sample_id(r): r for r in references}
    if len(by_id) != len(references):
        raise ValueError("Duplicate canonical margin IDs")
    missing = {r["sample_id"] for r in rows} - set(by_id)
    if missing:
        raise ValueError("Canonical margins lack expected samples: " + repr(sorted(missing)[:5]))
    for row in rows:
        ref = by_id[row["sample_id"]]
        if "question" in ref and ref["question"] != row["question"]:
            raise ValueError("Canonical margin question differs: " + row["sample_id"])
        answer = ref.get("gt_answer", ref.get("answer", ref.get("gt")))
        if answer is not None and str(answer).strip().rstrip(".").lower() != row["gt_answer"]:
            raise ValueError("Canonical margin label differs: " + row["sample_id"])
        if "margin" not in ref or not math.isfinite(float(ref["margin"])):
            raise ValueError("Canonical margins require finite 'margin' field")
    return by_id


def base_check(rows, result, backend, dataset, reference, contract):
    refs = reference_rows(reference, rows)
    disagreements = [r["sample_id"] for r in rows if (r["margin"] > 0) !=
                     (float(refs[r["sample_id"]]["margin"]) > 0)]
    expected = BASE_COUNTS[backend]
    counts_match = result["overall"]["correct"] == (expected[2] if dataset == "eventhallusion" else expected[0])
    if dataset == "videohallucer":
        counts_match = counts_match and result["both_correct_pairs"] == expected[1]
    passed = result["status"] == "COMPLETED" and counts_match and not disagreements
    return {"status": "PASS" if passed else "FAIL", "contract": contract,
            "expected_counts_match": counts_match, "prediction_disagreements": disagreements,
            "max_absolute_margin_difference": max(abs(r["margin"] - float(refs[r["sample_id"]]["margin"])) for r in rows),
            "reference_sha256": file_sha256(reference)}


def input_digest(batch):
    import hashlib
    import torch
    digest = hashlib.sha256()
    for key, value in sorted(batch.items()):
        digest.update(key.encode())
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def restore_margins(path, rows):
    predictions = []
    if not path.exists():
        return predictions
    offset = 0
    lines = path.read_bytes().splitlines(keepends=True)
    for i, line in enumerate(lines):
        # A final record is committed only once its newline has been written.
        if i == len(lines) - 1 and not line.endswith(b"\n"):
            with path.open("r+b") as f:
                f.truncate(offset)
            break
        try:
            pred = json.loads(line)
        except ValueError:
            raise ValueError("Corrupt committed margin history")
        if i >= len(rows) or any(pred[k] != rows[i][k] for k in (
                "sample_id", "pair_id", "question", "video_path", "gt_answer")):
            raise ValueError("Margin history differs from canonical samples")
        predictions.append(pred)
        offset += len(line)
    metrics(predictions, len(rows))
    return predictions


def evaluate(args, rows, contract):
    import signal
    import torch
    from video_tldpo.models.unified import answer_ids, answer_scores, load, prepare
    stopped = []
    signal.signal(signal.SIGTERM, lambda *_: stopped.append(True))
    deadline = min(time.time() + args.max_seconds, args.deadline_epoch or float("inf"))
    output = args.output / args.variant
    with locked_output(output):
        meta = output / "run_meta.json"
        if meta.exists() and json.loads(meta.read_text()) != contract:
            raise ValueError("Evaluation contract changed; use a new output directory")
        path = output / "margins.jsonl"
        if not meta.exists() and path.exists() and path.stat().st_size:
            raise ValueError("Margins exist without provenance")
        atomic_json(meta, contract)
        predictions = restore_margins(path, rows)
        if len(predictions) < len(rows):
            if not torch.cuda.is_available():
                raise RuntimeError("Unified evaluation requires a GPU job")
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            processor, model, tokenizer = load(args.model, args.backend)
            if args.variant == "adapter":
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
            model.eval()
            model.requires_grad_(False)
            tokens = answer_ids(tokenizer, args.backend)
            print("answer_token_ids:", tokens, flush=True)
            slowest = 0.0
            for row in rows[len(predictions):]:
                if stopped or time.time() + max(120, slowest * 1.5) >= deadline:
                    break
                started = time.monotonic()
                batch, indices = prepare(processor, model, row, args.backend)
                digest = input_digest(batch)
                with torch.inference_mode():
                    _, margin = answer_scores(model, batch, tokens)
                value = float(margin.item())
                pred = dict(row, margin=value, prediction="yes" if value > 0 else "no",
                            frame_indices=indices, input_digest=digest)
                pred["correct"] = pred["prediction"] == row["gt_answer"]
                if not math.isfinite(value):
                    raise FloatingPointError("Nonfinite margin")
                with path.open("a") as f:
                    f.write(json.dumps(pred, ensure_ascii=False) + "\n")
                    f.flush()
                    import os
                    os.fsync(f.fileno())
                predictions.append(pred)
                print("evaluated={}/{} id={} margin={:.6f}".format(len(predictions), len(rows), row["sample_id"], value), flush=True)
                del batch, margin
                slowest = max(slowest, time.monotonic() - started)
        result = metrics(predictions, len(rows))
        atomic_json(output / "metrics.json", result)
        if result["status"] != "COMPLETED":
            print("UNIFIED_EVALUATION_PARTIAL: rerun the same command", flush=True)
            return 3
        if args.variant == "base" and args.split == "test":
            check = base_check(predictions, result, args.backend, args.dataset, args.reference_margins, contract)
            atomic_json(output / "base_check.json", check)
            print("UNIFIED_BASE_CHECK:", check["status"], flush=True)
            if check["status"] != "PASS":
                return 4
        print("UNIFIED_{}_{}: PASS".format(args.dataset.upper(), args.variant.upper()), flush=True)
    if args.variant == "adapter":
        compare(args.output)
    return 0


def compare(root):
    root = Path(root)
    base_meta, adapter_meta = [json.loads((root / v / "run_meta.json").read_text()) for v in ("base", "adapter")]
    for key in ("protocol", "settings", "model_identity", "vh_data", "event_data", "backend", "split", "dataset", "runtime", "reference_sha256", "source_sha256"):
        if base_meta.get(key) != adapter_meta.get(key):
            raise ValueError("Base/adapter evaluation differs: " + key)
    base, adapter = [read_jsonl(root / v / "margins.jsonl") for v in ("base", "adapter")]
    for a, b in zip(base, adapter):
        if any(a[k] != b[k] for k in ("sample_id", "question", "gt_answer", "input_digest")):
            raise ValueError("Base/adapter inputs differ")
    expected = 409 if base_meta["dataset"] == "eventhallusion" else 310 if base_meta["split"] == "test" else 100
    bm, am = metrics(base, expected), metrics(adapter, expected)
    if bm["status"] != "COMPLETED" or am["status"] != "COMPLETED":
        raise ValueError("Comparison requires completed results")
    result = {"status": "COMPLETED", "protocol": PROTOCOL_ID, "records": expected,
              "base": bm, "adapter": am, "delta_accuracy_pp": 100 * (am["overall"]["accuracy"] - bm["overall"]["accuracy"]),
              "wrong_to_correct": sum(not a["correct"] and b["correct"] for a, b in zip(base, adapter)),
              "correct_to_wrong": sum(a["correct"] and not b["correct"] for a, b in zip(base, adapter))}
    atomic_json(root / "comparison.json", result)
    print(json.dumps(result, indent=2), flush=True)
