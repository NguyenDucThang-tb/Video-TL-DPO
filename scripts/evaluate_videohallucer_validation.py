"""Greedy, resumable base-vs-LoRA validation. Preflight needs stdlib only."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from video_tldpo.evaluation.video_validation import (
    PROTOCOL, append_prediction, atomic_json, build_metrics, compare_outputs,
    initialize_output, locked_output, parse_yes_no, restore_predictions,
    role_of, validate_inputs,
)


def tensor_input_digest(batch, torch):
    import hashlib
    digest = hashlib.sha256()
    for key in sorted(batch):
        digest.update(key.encode("utf-8"))
        value = batch[key]
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def print_comparison(root, dataset="videohallucer"):
    if dataset == "eventhallusion":
        from video_tldpo.evaluation.eventhallusion import compare_eventhallusion
        result = compare_eventhallusion(root / "base", root / "adapter", root / "comparison.json")
    else:
        result = compare_outputs(root / "base", root / "adapter", root / "comparison.json")
    print(json.dumps({
        "base_accuracy": result["base"]["overall"]["accuracy"],
        "adapter_accuracy": result["adapter"]["overall"]["accuracy"],
        "delta_accuracy_pp": result["delta_accuracy_pp"],
        "wrong_to_correct": result["wrong_to_correct"],
        "correct_to_wrong": result["correct_to_wrong"],
    }, indent=2), flush=True)
    label = "EVENTHALLUSION" if dataset == "eventhallusion" else "TEST" if result["split"] == "test" else "VALIDATION"
    print(label + "_COMPARISON: PASS", root / "comparison.json", flush=True)


def evaluate(args, rows, input_contract):
    from importlib.metadata import version
    import torch

    is_event = args.dataset == "eventhallusion"
    parser = parse_yes_no
    protocol = PROTOCOL
    metrics_builder = build_metrics
    if is_event:
        from video_tldpo.evaluation.eventhallusion import EVENT_PROTOCOL, eventhallusion_metrics, official_binary_parse
        parser, protocol, metrics_builder = official_binary_parse, EVENT_PROTOCOL, eventhallusion_metrics
    if args.backend != "qwen25":
        from video_tldpo.models.vlm_backends import backend_spec
        protocol = dict(protocol, backend_frame_protocol=backend_spec(args.backend)["protocol"])
    contract = {
        "variant": args.variant, "inputs": input_contract, "protocol": protocol,
        "runtime": {name: version(name) for name in ("torch", "transformers", "peft", "decord")},
        "backend": args.backend,
    }
    output = args.output / args.variant
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    deadline = time.time() + args.max_seconds
    if args.deadline_epoch is not None:
        deadline = min(deadline, args.deadline_epoch)

    with locked_output(output):
        initialize_output(output, contract)
        path = output / "predictions.jsonl"
        predictions = restore_predictions(path, rows, parser=parser)
        if is_event:
            for prediction, row in zip(predictions, rows):
                if any(prediction.get(key) != row[key] for key in ("category", "video_id", "question_type")):
                    raise ValueError("EventHallusion resume metadata changed")
        print("variant:", args.variant, "resume_predictions:", len(predictions), flush=True)
        if len(predictions) < len(rows):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; inference belongs in a GPU job")
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            if args.backend == "qwen25":
                from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration
                from train_videohallucer_vanilla_resumable import make_inputs
                processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True)
                prepare_inputs = lambda row: make_inputs(processor, row)
            else:
                from video_tldpo.models.vlm_backends import load_backend, prepare, generate
                processor, model, _ = load_backend(args.model, args.backend, trainable=False)
                prepare_inputs = lambda row: prepare(processor, row, args.backend, torch.device("cpu"))
            if args.variant == "adapter":
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
            model.eval()
            model.requires_grad_(False)
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = True
            if args.backend == "internvl35" and hasattr(model, "language_model") \
                    and hasattr(model.language_model.config, "use_cache"):
                model.language_model.config.use_cache = True
            if args.backend == "qwen25":
                generation = GenerationConfig(
                    max_new_tokens=PROTOCOL["max_new_tokens"], do_sample=False, num_beams=1,
                    temperature=1.0, top_p=1.0, top_k=50, use_cache=True,
                    eos_token_id=model.generation_config.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    bos_token_id=processor.tokenizer.bos_token_id,
                )
            slowest_step = 0.0
            for row in rows[len(predictions):]:
                if stop or time.time() + max(120.0, slowest_step * 1.5) >= deadline:
                    break
                started = time.monotonic()
                print("processing:", row["sample_id"], flush=True)
                if args.backend == "qwen25":
                    prompt, encode = prepare_inputs(row)
                    cpu_inputs = encode(prompt, torch.device("cpu"))
                else:
                    prepared = prepare_inputs(row)
                    cpu_inputs = prepared.encode(prepared.prompt, torch.device("cpu"))
                duration_diagnostic = None
                if is_event:
                    if args.backend == "qwen25":
                        from video_tldpo.evaluation.eventhallusion import validate_duration
                        duration_diagnostic = validate_duration(row, cpu_inputs, strict=False)
                    else:
                        expected = row.get("duration_seconds")
                        if expected is not None:
                            observed = prepared.source_frame_count / prepared.source_fps
                            difference = abs(observed - expected)
                            tolerance = max(1.5, 0.1 * expected)
                            duration_diagnostic = {"annotated_seconds": expected,
                                "decoded_seconds": observed,
                                "absolute_difference_seconds": difference,
                                "tolerance_seconds": tolerance,
                                "within_tolerance": difference <= tolerance}
                    if duration_diagnostic and not duration_diagnostic["within_tolerance"]:
                        print("duration_metadata_difference: {} {}".format(
                            row["video_id"], json.dumps(duration_diagnostic, sort_keys=True)), flush=True)
                digest = tensor_input_digest(cpu_inputs, torch)
                inputs = {key: value.to("cuda:0") if hasattr(value, "to") else value
                          for key, value in cpu_inputs.items()}
                if args.backend == "qwen25":
                    with torch.inference_mode():
                        generated = model.generate(**inputs, generation_config=generation)
                    answer_ids = generated[:, inputs["input_ids"].shape[1]:]
                    answer = processor.batch_decode(
                        answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    generated_count = int(answer_ids.shape[1])
                else:
                    answer, generated_count = generate(model, processor, prepared, args.backend,
                                                       max_new_tokens=PROTOCOL["max_new_tokens"])
                    answer_ids = None
                parsed = parser(answer)
                prediction = {key: row[key] for key in ("sample_id", "pair_id", "subset", "video_path", "question")}
                prediction.update({"role": row["role"] if is_event else role_of(row), "gt_answer": row["gt_answer"].strip().lower(),
                                   "raw_answer": answer, "prediction": parsed,
                                   "correct": parsed == row["gt_answer"].strip().lower(),
                                   "input_digest": digest,
                                   "generated_tokens": generated_count,
                                   "reached_token_limit": generated_count >= PROTOCOL["max_new_tokens"]})
                if is_event:
                    prediction.update({key: row[key] for key in ("video_id", "category", "question_type")})
                    prediction["strict_prediction"] = parse_yes_no(answer)
                    prediction["duration_diagnostic"] = duration_diagnostic
                append_prediction(path, prediction)
                predictions.append(prediction)
                slowest_step = max(slowest_step, time.monotonic() - started)
                print("evaluated={}/{} answer={!r} prediction={} correct={}".format(
                    len(predictions), len(rows), answer, parsed, prediction["correct"]), flush=True)
                del inputs, cpu_inputs
                if args.backend == "qwen25":
                    del generated, answer_ids, encode
                else:
                    del prepared, answer_ids

        metrics = metrics_builder(predictions, len(rows))
        atomic_json(output / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        if metrics["status"] != "COMPLETED":
            print(("EVENTHALLUSION" if is_event else args.split.upper()) + "_PARTIAL: saved predictions; rerun the same evaluation command", flush=True)
            return 3
        label = "EVENTHALLUSION" if is_event else "TEST" if args.split == "test" else "VALIDATION"
        print("{}_{}: PASS".format(label, args.variant.upper()), flush=True)

    if args.variant == "adapter":
        print_comparison(args.output, args.dataset)
    return 0


def main():
    project = Path(os.environ.get("VIDEO_TLDPO_PROJECT", "/g/data/hn98/dd9648/Video-TL-DPO"))
    training = project / "outputs/vanilla_dpo_qwen25_vh_continuous_v2"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("videohallucer", "eventhallusion"),
                        default=os.environ.get("VIDEO_TLDPO_EVAL_DATASET", "videohallucer"))
    parser.add_argument("--backend", choices=("qwen25", "qwen3", "internvl35"),
                        default=os.environ.get("VIDEO_TLDPO_BACKEND", "qwen25"))
    parser.add_argument("--dataset-root", type=Path, default=Path(os.environ.get(
        "VIDEO_TLDPO_EVENT_ROOT", "/scratch/jp09/dd9648/datasets_video_hallu/eventhallusion")))
    parser.add_argument("--questions-dir", type=Path, default=Path(os.environ["VIDEO_TLDPO_EVENT_QUESTIONS"])
                        if os.environ.get("VIDEO_TLDPO_EVENT_QUESTIONS") else None)
    parser.add_argument("--split", choices=("val", "test"), default=os.environ.get("VIDEO_TLDPO_EVAL_SPLIT", "val"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--validation-data", type=Path, default=project / "artifacts/val_binary.jsonl")
    parser.add_argument("--train-data", type=Path, default=project / "artifacts/train_binary.jsonl")
    parser.add_argument("--manifest", type=Path, default=project / "artifacts/split_manifest.json")
    parser.add_argument("--model", type=Path, default=Path(os.environ.get(
        "VIDEO_TLDPO_MODEL", "/g/data/hn98/dd9648/models/Qwen2.5-VL-7B-Instruct")))
    parser.add_argument("--adapter", type=Path, default=training / "trainer_state/latest/adapter")
    parser.add_argument("--train-summary", type=Path, default=training / "train_summary.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variant", choices=("base", "adapter"), default="base")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=2700)
    parser.add_argument("--deadline-epoch", type=float)
    args = parser.parse_args()
    args.data = args.data or project / ("artifacts/test_records.jsonl" if args.split == "test" else "artifacts/val_binary.jsonl")
    if args.output is None:
        folder = "eventhallusion_vanilla_dpo_v2" if args.dataset == "eventhallusion" else \
            "test_vanilla_dpo_v1" if args.split == "test" else "validation_vanilla_dpo_v1"
        args.output = project / "outputs" / folder
    if args.compare_only:
        print_comparison(args.output, args.dataset)
        return 0
    if args.max_seconds <= 120:
        parser.error("max-seconds must exceed 120")
    try:
        if args.dataset == "eventhallusion":
            from video_tldpo.evaluation.eventhallusion import validate_eventhallusion_inputs
            rows, inputs = validate_eventhallusion_inputs(args.dataset_root, args.train_data, args.model,
                                                         args.adapter, args.train_summary, args.questions_dir,
                                                         args.backend)
        else:
            rows, inputs = validate_inputs(args.data, args.train_data, args.model, args.adapter,
                                          args.train_summary, args.manifest, args.split, args.validation_data,
                                          args.backend)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, "EVALUATION PREFLIGHT FAILED: " + str(exc) + "\n")
    print(("eventhallusion" if args.dataset == "eventhallusion" else args.split) + "_preflight: PASS",
          json.dumps(inputs["counts"], sort_keys=True), flush=True)
    if args.preflight_only:
        print("preflight_only: no GPU work or jobs submitted", flush=True)
        return 0
    return evaluate(args, rows, inputs)


if __name__ == "__main__":
    sys.exit(main())
