"""Score a trained adapter on fixed VH test questions with all video pixels black."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from video_tldpo.data.unified import load_vh
from video_tldpo.evaluation.unified import metrics
from video_tldpo.evaluation.video_validation import atomic_json, file_sha256, read_jsonl
from video_tldpo.models.unified import answer_ids, answer_scores, load, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qwen3", "internvl35"), required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Completed language_attention_v1 run root")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--canonical-split", type=Path, required=True)
    parser.add_argument("--canonical-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    splits, data = load_vh(args.project / "artifacts", args.canonical_split,
                           args.canonical_eval, canonical_train=None)
    rows = splits["test"]
    train_root = args.run_root / args.backend / "train"
    summary = json.loads((train_root / "train_summary.json").read_text())
    adapter = train_root / "trainer_state/latest/adapter"
    if summary.get("status") != "COMPLETED" or summary.get("records") != 300:
        raise ValueError("Expected completed 300-record language_attention adapter")
    if summary.get("contract", {}).get("settings", {}).get("lora_scope") != "language_attention":
        raise ValueError("Selected adapter was not trained with language_attention scope")
    if Path(summary["adapter_checkpoint"]).resolve() != adapter.resolve():
        raise ValueError("Adapter path differs from completed training summary")

    processor, base, tokenizer = load(args.model, args.backend)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    model.eval().requires_grad_(False)
    tokens = answer_ids(tokenizer, args.backend)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    margins_path = output / "margins.jsonl"
    predictions = []
    import torch
    with margins_path.open("w") as stream:
        for index, row in enumerate(rows, 1):
            batch, indices = prepare(processor, model, row, args.backend, black_frames=True)
            with torch.inference_mode():
                _, margin = answer_scores(model, batch, tokens)
            value = float(margin.item())
            pred = dict(row, margin=value, prediction="yes" if value > 0 else "no",
                        correct=("yes" if value > 0 else "no") == row["gt_answer"],
                        frame_indices=indices, visual_input="all_black_rgb_448")
            stream.write(json.dumps(pred, ensure_ascii=False) + "\n")
            predictions.append(pred)
            if index % 25 == 0 or index == len(rows):
                print("scored={}/{}".format(index, len(rows)), flush=True)
            del batch
    result = metrics(predictions, len(rows))
    normal_path = args.run_root / args.backend / "videohallucer_test/adapter/margins.jsonl"
    normal = read_jsonl(normal_path)
    if any(any(left[key] != right[key] for key in ("sample_id", "question", "gt_answer", "video_path"))
           for left, right in zip(normal, predictions)) or len(normal) != len(predictions):
        raise ValueError("Black-frame rows differ from normal adapter test rows")
    normal_metrics = metrics(normal, len(rows))
    comparison = {
        "status": result["status"], "backend": args.backend,
        "adapter_scope": "language_attention", "visual_input": "all_black_rgb_448",
        "records": len(rows), "normal_adapter": normal_metrics, "black_frame_adapter": result,
        "delta_black_minus_normal_pp": {
            "VH_Q_acc": 100 * (result["overall"]["accuracy"] - normal_metrics["overall"]["accuracy"]),
            "VH_pair_acc": 100 * (result["pair_accuracy"] - normal_metrics["pair_accuracy"]),
        },
        "adapter_sha256": file_sha256(adapter / "adapter_model.safetensors"),
        "normal_margin_sha256": file_sha256(normal_path),
        "split_manifest_sha256": data["canonical_split"],
        "evaluation_source_sha256": __import__("video_tldpo.unified", fromlist=["source_fingerprint"]).source_fingerprint(),
    }
    atomic_json(output / "metrics.json", result)
    atomic_json(output / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2), flush=True)
    print("BLACK_FRAME_ADAPTER_EVALUATION: PASS", flush=True)


if __name__ == "__main__":
    main()
