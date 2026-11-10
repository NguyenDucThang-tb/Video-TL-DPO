"""Gadi entry point for the unified protocol; preflight imports only stdlib."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from video_tldpo.data.unified import load_vh
from video_tldpo.evaluation.video_validation import file_sha256
from video_tldpo.models.unified import identity
from video_tldpo.unified import MODEL_NAMES, PROTOCOL_ID, SETTINGS, source_fingerprint


def main(argv=None):
    user = os.environ.get("USER", "dd9648")
    home = Path("/g/data/hn98") / user
    project = Path(os.environ.get("VIDEO_TLDPO_PROJECT", str(home / "Video-TL-DPO")))
    canonical = home / "OPA-DPO-Video/outputs"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("train", "evaluate"))
    p.add_argument("--backend", choices=tuple(MODEL_NAMES), required=True)
    p.add_argument("--project", type=Path, default=project)
    p.add_argument("--model", type=Path)
    p.add_argument("--artifacts", type=Path)
    p.add_argument("--canonical-split", type=Path, default=Path(os.environ.get("VIDEO_TLDPO_CANONICAL_SPLIT", str(canonical / "data/split.json"))))
    p.add_argument("--canonical-eval", type=Path, default=Path(os.environ.get("VIDEO_TLDPO_CANONICAL_EVAL", str(canonical / "data/eval_vh.jsonl"))))
    p.add_argument("--canonical-train", type=Path)
    p.add_argument("--dataset", choices=("videohallucer", "eventhallusion"), default="videohallucer")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--variant", choices=("base", "adapter"), default="base")
    p.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("VIDEO_TLDPO_EVENT_ROOT", "/scratch/hn98/{}/runs/exp1/EventHallusion".format(user))))
    p.add_argument("--questions-dir", type=Path)
    p.add_argument("--reference-margins", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--train-summary", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--max-seconds", type=int, default=2700)
    p.add_argument("--deadline-epoch", type=float)
    p.add_argument("--save-every", type=int, default=5, help="optimizer steps, not sample count")
    a = p.parse_args(argv)
    a.model = a.model or home / "models" / MODEL_NAMES[a.backend]
    a.artifacts = a.artifacts or a.project / "artifacts"
    a.canonical_train = a.canonical_train or Path(os.environ.get(
        "VIDEO_TLDPO_CANONICAL_TRAIN", str(a.artifacts / "train_binary.jsonl")))
    root = a.project / "outputs" / PROTOCOL_ID / a.backend
    trainout = root / "train"
    a.output = a.output or (trainout if a.command == "train" else root / (a.dataset + "_" + a.split))
    a.adapter = a.adapter or trainout / "trainer_state/latest/adapter"
    a.train_summary = a.train_summary or trainout / "train_summary.json"
    refname = {"qwen25": "base", "qwen3": "base_qwen3", "internvl35": "base_internvl"}[a.backend]
    if a.reference_margins is None:
        candidate = canonical / "eval" / refname / "margins.jsonl"
        a.reference_margins = candidate if candidate.is_file() else None
    if a.max_seconds <= 120 or a.save_every < 1:
        p.error("max-seconds must exceed 120; save-every must be positive")
    try:
        needs_train_source = a.command == "train" or (a.command == "evaluate" and a.variant == "adapter")
        splits, data = load_vh(a.artifacts, a.canonical_split, a.canonical_eval,
                               canonical_train=a.canonical_train if needs_train_source else None)
        canonical_train_sha256 = data.pop("canonical_train", None)
        contract = {"protocol": PROTOCOL_ID, "settings": SETTINGS, "backend": a.backend,
                    "model_identity": identity(a.model, a.backend), "vh_data": data,
                    "reference_sha256": file_sha256(a.reference_margins) if a.reference_margins is not None else None,
                    "source_sha256": source_fingerprint()}
        if canonical_train_sha256 is not None:
            contract["canonical_train_sha256"] = canonical_train_sha256
        if a.command == "train":
            rows = splits["train"]
        else:
            contract.update(dataset=a.dataset, split=a.split, variant=a.variant)
            if a.dataset == "eventhallusion":
                from video_tldpo.evaluation.eventhallusion import load_eventhallusion
                rows, event = load_eventhallusion(a.dataset_root, a.questions_dir)
                if event["counts"]["eventhallusion_records"] != 409:
                    raise ValueError("Unified EventHallusion requires all 409 questions")
                contract["event_data"] = event
            else:
                rows = splits[a.split]
            if a.split == "test" and a.reference_margins is not None:
                from video_tldpo.evaluation.unified import reference_rows
                reference_rows(a.reference_margins, rows)
            if a.variant == "adapter":
                summary = json.loads(a.train_summary.read_text())
                saved_contract = summary.get("contract", {})
                if summary.get("status") != "COMPLETED" or summary.get("records") != 300 or summary.get("optimizer_steps") != 38:
                    raise ValueError("Adapter requires completed unified 300-record/38-step training")
                for key in ("protocol", "settings", "model_identity", "vh_data", "backend", "source_sha256", "reference_sha256", "canonical_train_sha256"):
                    if saved_contract.get(key) != contract[key]:
                        raise ValueError("Adapter training differs from unified configuration: " + key)
                if Path(summary["adapter_checkpoint"]).resolve() != a.adapter.resolve():
                    raise ValueError("Adapter path differs from completed summary")
                contract["adapter_sha256"] = file_sha256(a.adapter / "adapter_model.safetensors")
                contract["adapter_config_sha256"] = file_sha256(a.adapter / "adapter_config.json")
        print("UNIFIED_PREFLIGHT: PASS backend={} command={} records={}".format(a.backend, a.command, len(rows)), flush=True)
        if a.preflight_only:
            return 0
        from importlib.metadata import version
        contract["runtime"] = {name: version(name) for name in ("torch", "transformers", "peft", "decord", "numpy", "pillow", "torchvision")}
        if a.backend in ("qwen3", "internvl35") and contract["runtime"]["transformers"] != "4.57.6":
            raise ValueError("Unified Qwen3/InternVL requires transformers 4.57.6")
        if a.command == "train":
            from video_tldpo.training.unified import train
            return train(a, rows, contract)
        from video_tldpo.evaluation.unified import evaluate
        return evaluate(a, rows, contract)
    except (OSError, ValueError, KeyError) as exc:
        p.exit(1, "UNIFIED FAILED: {}\n".format(exc))


def legacy_main(command, backend=None):
    """Route the historic runner names to this protocol, including --data paths."""
    argv = list(sys.argv[1:])
    if "--data" in argv:
        index = argv.index("--data")
        argv[index:index + 2] = ["--artifacts", str(Path(argv[index + 1]).parent)]
    if backend is not None and "--backend" not in argv:
        argv = ["--backend", backend] + argv
    print("Using unified_config.md protocol; old output/checkpoint contracts will be rejected.", flush=True)
    return main([command] + argv)


if __name__ == "__main__":
    sys.exit(main())
