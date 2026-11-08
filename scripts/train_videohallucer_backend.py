"""Resumable Vanilla DPO runner for Qwen3-VL and InternVL3.5."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import torch
import torch.nn.functional as F

from video_tldpo.models.vlm_backends import (
    backend_spec, language_lora_targets, load_backend, model_identity,
    prepare, response_logp,
)
import train_videohallucer_vanilla_resumable as common


def contract(data: Path, model: Path, backend: str):
    from importlib.metadata import version
    SETTINGS = {**common.SETTINGS, "backend": backend, "video_protocol": backend_spec(backend)["protocol"],
                "lora_scope": "language_model_only_exact_linear_names"}
    packages = ["torch", "transformers", "peft", "decord", "numpy", "pillow", "torchvision"]
    if backend == "internvl35":
        packages += ["timm", "einops"]
    return {"schema": 1, "settings": SETTINGS,
            "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            "model_identity": model_identity(model, backend),
            "versions": {name: version(name) for name in packages}}


def train(args):
    started = time.monotonic()
    deadline = time.time() + args.max_seconds
    if args.deadline_epoch is not None:
        deadline = min(deadline, args.deadline_epoch)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    rows = common.load_rows(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.output / "trainer_state"
    run_contract = contract(args.data, args.model, args.backend)
    checkpoint = common.find_checkpoint(checkpoint_root, run_contract, len(rows))
    if checkpoint is not None and checkpoint.name == "checkpoint.tmp":
        checkpoint = common.promote_temporary(checkpoint_root)
    saved = torch.load(checkpoint / "trainer.pt", map_location="cpu", weights_only=False) if checkpoint else None
    state0 = saved["state"] if saved else {"next_record_index": 0, "losses": [], "contract": run_contract}
    if state0["next_record_index"] == len(rows):
        common.write_summary(args.output, state0, len(rows), checkpoint)
        print("training_already_complete:", len(rows), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if time.time() >= deadline:
        raise RuntimeError("Environment setup exhausted the job budget")

    from peft import LoraConfig, PeftModel, get_peft_model
    torch.manual_seed(common.SETTINGS["seed"])
    torch.cuda.manual_seed_all(common.SETTINGS["seed"])
    print("backend:", args.backend, "gpu:", torch.cuda.get_device_name(0), flush=True)
    print("records:", len(rows), "model_identity:", json.dumps(run_contract["model_identity"], sort_keys=True), flush=True)

    processor, policy_base, _ = load_backend(args.model, args.backend, trainable=True)
    if checkpoint is None:
        targets = language_lora_targets(policy_base, args.backend)
        lora = LoraConfig(r=common.SETTINGS["lora_r"], lora_alpha=common.SETTINGS["lora_alpha"],
                          lora_dropout=common.SETTINGS["lora_dropout"], bias="none",
                          task_type="CAUSAL_LM" if args.backend == "qwen3" else None,
                          target_modules=targets)
        policy = get_peft_model(policy_base, lora)
        print("lora_target_count:", len(targets), "lora_language_only: PASS", flush=True)
    else:
        policy = PeftModel.from_pretrained(policy_base, checkpoint / "adapter", is_trainable=True)
    if hasattr(policy, "enable_input_require_grads"):
        policy.enable_input_require_grads()
    policy.train()

    _, reference, _ = load_backend(args.model, args.backend, trainable=False)
    reference.eval()
    reference.requires_grad_(False)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters")
    optimizer = torch.optim.AdamW(trainable, lr=common.SETTINGS["learning_rate"])
    if checkpoint:
        names = [name for name, p in policy.named_parameters() if p.requires_grad]
        if names != saved["trainable_names"]:
            raise RuntimeError("Trainable parameter names changed across resume")
        optimizer.load_state_dict(saved["optimizer"])
        common.restore_rng(saved["rng"])
        del saved

    next_index = int(state0["next_record_index"])
    losses = list(state0.get("losses", []))
    last_saved = next_index
    slowest = 0.0
    for index in range(next_index, len(rows)):
        if stop or time.time() + max(120.0, 1.5 * slowest) >= deadline:
            break
        t0 = time.monotonic()
        row = rows[index]
        print("processing:", row["sample_id"], flush=True)
        prepared = prepare(processor, row, args.backend, torch.device("cpu"))
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            rc = response_logp(reference, processor, prepared, row["chosen"], False)
            rr = response_logp(reference, processor, prepared, row["rejected"], False)
        pc = response_logp(policy, processor, prepared, row["chosen"], True)
        pr = response_logp(policy, processor, prepared, row["rejected"], True)
        loss = -F.logsigmoid(common.SETTINGS["beta"] * ((pc - rc) - (pr - rr)))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite DPO loss at " + row["sample_id"])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        if norm.item() <= 0:
            raise RuntimeError("LoRA gradient is zero")
        if any(p.grad is not None for p in reference.parameters()):
            raise RuntimeError("Frozen reference received gradients")
        optimizer.step()
        losses.append(float(loss.item()))
        next_index = index + 1
        slowest = max(slowest, time.monotonic() - t0)
        print("record={}/{} loss={:.8f} grad_norm={:.6f} frames={} step_seconds={:.1f}".format(
            next_index, len(rows), loss.item(), norm.item(), len(prepared.frame_indices), slowest), flush=True)
        del loss, pc, pr, rc, rr, prepared
        if next_index % args.save_every == 0:
            state = {"next_record_index": next_index, "losses": losses, "contract": run_contract}
            common.save_checkpoint(policy, optimizer, checkpoint_root, state)
            last_saved = next_index

    state = {"next_record_index": next_index, "losses": losses, "contract": run_contract}
    if next_index != last_saved or checkpoint is None:
        common.save_checkpoint(policy, optimizer, checkpoint_root, state)
    if next_index == len(rows):
        common.write_summary(args.output, state, len(rows))
        print("VANILLA_DPO_{}_TRAINING: PASS".format(args.backend.upper()), flush=True)
    else:
        print("TRAINING_PAUSED_FOR_NEXT_JOB: next_record_index={} elapsed_seconds={:.1f}".format(
            next_index, time.monotonic() - started), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("qwen3", "internvl35"), required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=2700)
    p.add_argument("--deadline-epoch", type=float)
    p.add_argument("--save-every", type=int, default=5)
    a = p.parse_args()
    if a.max_seconds <= 120 or a.save_every < 1:
        p.error("max-seconds must exceed 120; save-every must be positive")
    with common.output_lock(a.output):
        train(a)


if __name__ == "__main__":
    main()
