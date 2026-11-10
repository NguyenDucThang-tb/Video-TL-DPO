"""Legacy artifact/scoring and atomic-checkpoint helpers.

LEGACY_SETTINGS describes historical r8/lr1e-5 artifacts only. The command-line
entry point routes to run_unified.py; new settings live in video_tldpo.unified.
The atomic checkpoint helpers are also reused by the unified trainer.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
import random
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCHEMA = 2
LEGACY_SETTINGS = {"frames": 8, "beta": 0.1, "learning_rate": 1e-5,
            "seed": 0, "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
            "sampled_fps_forwarded": True, "response_eos": False}


@contextmanager
def output_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".train.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another trainer is using {output}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def run_contract(data: Path, model: Path):
    return {
        "schema": SCHEMA, "settings": LEGACY_SETTINGS,
        "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "model_path": str(model.resolve()),
        "model_files": {name: hashlib.sha256((model / name).read_bytes()).hexdigest()
                        for name in ("config.json", "model.safetensors.index.json", "preprocessor_config.json")},
        "versions": {name: version(name) for name in
                     ("torch", "transformers", "peft", "qwen-vl-utils", "numpy")},
    }


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 300:
        raise RuntimeError(f"Expected exactly 300 train records, found {len(rows)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("Duplicate sample_id in training JSONL")
    for row in rows:
        if row.get("split") != "train":
            raise RuntimeError(f"Non-train record: {row['sample_id']}")
        for key in ("video_path", "question", "chosen", "rejected"):
            if not row.get(key):
                raise RuntimeError(f"Missing {key} in {row.get('sample_id')}")
        if not Path(row["video_path"]).is_file():
            raise FileNotFoundError(row["video_path"])
        chosen = row["chosen"].strip().rstrip(".").lower()
        rejected = row["rejected"].strip().rstrip(".").lower()
        if {chosen, rejected} != {"yes", "no"} or chosen != row["gt_answer"].strip().lower():
            raise RuntimeError(f"Invalid binary preference: {row['sample_id']}")
    return rows


def video_sampling_spec(video_path: str):
    """Request 8 uniform frames, capped to the clip's available even count.

    qwen-vl-utils clamps the FPS/min/max route to the available frame count;
    its explicit nframes route raises when a clip contains fewer frames.
    For clips with at least 8 frames, both routes select the same 8 indices.
    """
    return {"type": "video", "video": video_path, "fps": 2.0,
            "min_frames": LEGACY_SETTINGS["frames"], "max_frames": LEGACY_SETTINGS["frames"]}


def make_inputs(processor, row: dict):
    from qwen_vl_utils import process_vision_info
    messages = [{
        "role": "user",
        "content": [
            video_sampling_spec(row["video_path"]),
            {"type": "text", "text": row["question"] + " Answer yes or no."},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    fps = video_kwargs.get("fps")
    values = fps if isinstance(fps, (list, tuple)) else [fps]
    if not values or not all(x is not None and math.isfinite(float(x)) and float(x) > 0 for x in values):
        raise RuntimeError("Video reader returned invalid sampled FPS")
    sampled_frames = int(video_inputs[0].shape[0])
    if not 2 <= sampled_frames <= LEGACY_SETTINGS["frames"] or sampled_frames % 2:
        raise RuntimeError(f"Invalid sampled video frame count: {sampled_frames}")
    if sampled_frames < LEGACY_SETTINGS["frames"]:
        print("short_video_sampling:", row["video_path"],
              f"sampled_frames={sampled_frames}", f"sampled_fps={fps}", flush=True)

    def process(text):
        batch = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        required = {"input_ids", "attention_mask", "pixel_values_videos",
                    "video_grid_thw", "second_per_grid_ts"}
        if not required.issubset(batch):
            raise RuntimeError(f"Missing native video inputs: {required - set(batch)}")
        return batch

    prompt_batch = process(prompt)
    prompt_ids = prompt_batch["input_ids"][0].tolist()
    cache = {prompt: prompt_batch}

    def encode(text: str, device: torch.device):
        if text not in cache:
            batch = process(text)
            if batch["input_ids"][0].tolist()[:len(prompt_ids)] != prompt_ids:
                raise RuntimeError("Prompt/response token boundary changed")
            cache[text] = batch
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in cache[text].items()
        }

    return prompt, encode


def response_suffix_logp(logits, input_ids, count):
    if count <= 0 or count >= input_ids.shape[1]:
        raise RuntimeError("Invalid response token count")
    selected = logits[:, -(count + 1):-1, :].float()
    labels = input_ids[:, -count:]
    return F.log_softmax(selected, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1).sum()


def response_logp(model, processor, prompt, encode, response, with_grad):
    device = next(model.parameters()).device
    batch = encode(prompt + response, device)
    response_ids = processor.tokenizer(
        response, add_special_tokens=False
    )["input_ids"]
    if not response_ids:
        raise RuntimeError("Response tokenized to an empty sequence")
    if batch["input_ids"][0].tolist()[-len(response_ids):] != response_ids:
        raise RuntimeError("Response token alignment check failed")

    grad_context = torch.enable_grad() if with_grad else torch.no_grad()
    with grad_context:
        logits = model(**batch, use_cache=False).logits
        return response_suffix_logp(logits, batch["input_ids"], len(response_ids))


def save_checkpoint(policy, optimizer, checkpoint_root: Path, state: dict, scheduler=None) -> None:
    """Keep two checkpoint generations and allow interrupted promotion recovery."""
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_root / "checkpoint.tmp"
    latest = checkpoint_root / "latest"

    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    policy.save_pretrained(temporary / "adapter")
    payload = {"optimizer": optimizer.state_dict(), "state": state,
               "scheduler": scheduler.state_dict() if scheduler is not None else None,
               "rng": capture_rng(),
               "trainable_names": [name for name, p in policy.named_parameters() if p.requires_grad]}
    with (temporary / "trainer.pt").open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    # This commit marker is written only after all adapter/trainer files exist.
    (temporary / "state.json").write_text(json.dumps(state, indent=2))
    promote_temporary(checkpoint_root)
    print(
        "checkpoint_saved:",
        str(latest),
        "next_record_index:",
        state["next_record_index"],
        flush=True,
    )


def promote_temporary(checkpoint_root: Path) -> Path:
    temporary = checkpoint_root / "checkpoint.tmp"
    latest, previous = checkpoint_root / "latest", checkpoint_root / "previous"
    if latest.exists():
        if previous.exists():
            shutil.rmtree(previous)
        os.replace(latest, previous)
    os.replace(temporary, latest)
    return latest


def find_checkpoint(checkpoint_root: Path, contract: dict, records: int) -> Path | None:
    candidates = []
    for name in ("latest", "previous", "checkpoint.tmp"):
        path = checkpoint_root / name
        if not (path / "state.json").is_file():
            if name != "checkpoint.tmp" and (path / "trainer.pt").is_file():
                raise RuntimeError("Legacy checkpoint lacks run metadata; preserve it and use a new output directory")
            continue
        try:
            state = json.loads((path / "state.json").read_text())
        except (OSError, ValueError) as exc:
            print("invalid_checkpoint_metadata:", path, str(exc), flush=True)
            continue
        if state.get("contract") != contract:
            raise RuntimeError("Checkpoint dataset/model/settings differ from this run")
        cursor = state["next_record_index"]
        if type(cursor) is not int or not 0 <= cursor <= records:
            raise RuntimeError("Checkpoint record cursor is invalid")
        if len(state["losses"]) != cursor or not all(math.isfinite(x) for x in state["losses"]):
            raise RuntimeError("Checkpoint losses do not match its record cursor")
        candidates.append((cursor, path, state))
    for _, path, metadata in sorted(candidates, key=lambda item: item[0], reverse=True):
        try:
            payload = torch.load(path / "trainer.pt", map_location="cpu", weights_only=False)
            if payload["state"] != metadata:
                raise ValueError("Trainer state differs from commit marker")
            for key in ("rng", "optimizer", "trainable_names"):
                if key not in payload:
                    raise ValueError(f"Missing checkpoint field: {key}")
            if state["contract"].get("protocol") == "unified_adaptive_seed_v2" and payload.get("scheduler") is None:
                raise ValueError("Unified checkpoint lacks scheduler state")
            if not (path / "adapter/adapter_model.safetensors").is_file():
                raise ValueError("Adapter weights are missing")
            return path
        except (OSError, EOFError, ValueError, KeyError, RuntimeError) as exc:
            print("invalid_checkpoint:", path, str(exc), flush=True)
    if candidates or any((checkpoint_root / name).exists() for name in ("latest", "previous")):
        raise RuntimeError("No complete checkpoint can be restored; refusing to restart at record 0")
    return None


def write_summary(output, state, records, checkpoint=None):
    losses = state["losses"]
    temporary = output / "train_summary.json.tmp"
    temporary.write_text(json.dumps({
        "status": "COMPLETED", "records": records, "frames": LEGACY_SETTINGS["frames"],
        "frame_policy": "uniform_up_to_requested_even",
        "contract": state["contract"],
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_min": min(losses), "loss_max": max(losses),
        "adapter_checkpoint": str((checkpoint or output / "trainer_state/latest") / "adapter"),
    }, indent=2))
    os.replace(temporary, output / "train_summary.json")


def train(args) -> None:
    started = time.monotonic()
    deadline = time.time() + args.max_seconds
    if args.deadline_epoch is not None:
        deadline = min(deadline, args.deadline_epoch)
    stop_requested = []
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.append(True))
    rows = load_rows(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.output / "trainer_state"
    contract = run_contract(args.data, args.model)
    checkpoint = find_checkpoint(checkpoint_root, contract, len(rows))
    if checkpoint is not None and checkpoint.name == "checkpoint.tmp":
        checkpoint = promote_temporary(checkpoint_root)
    saved = torch.load(checkpoint / "trainer.pt", map_location="cpu", weights_only=False) if checkpoint else None
    saved_state = saved["state"] if saved else {"next_record_index": 0, "losses": [], "contract": contract}
    if saved_state["next_record_index"] == len(rows):
        write_summary(args.output, saved_state, len(rows), checkpoint)
        print("training_already_complete:", len(rows), flush=True)
        return
    if (args.output / "train_summary.json").exists():
        raise RuntimeError("Completion summary exists without a matching completed checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if time.time() >= deadline:
        raise RuntimeError("Environment setup exhausted the job budget")

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    random.seed(LEGACY_SETTINGS["seed"])
    np.random.seed(LEGACY_SETTINGS["seed"])
    torch.manual_seed(LEGACY_SETTINGS["seed"])
    torch.cuda.manual_seed_all(LEGACY_SETTINGS["seed"])

    print("gpu:", torch.cuda.get_device_name(0), flush=True)
    print("records:", len(rows), flush=True)
    print("output:", args.output, flush=True)

    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, use_fast=False
    )
    policy_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )
    policy_base.config.use_cache = False
    policy_base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if checkpoint is None:
        lora_config = LoraConfig(
            r=LEGACY_SETTINGS["lora_r"],
            lora_alpha=LEGACY_SETTINGS["lora_alpha"],
            lora_dropout=LEGACY_SETTINGS["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        policy = get_peft_model(policy_base, lora_config)
        print("resume_checkpoint: none; starting at record 0", flush=True)
    else:
        print("resume_checkpoint:", checkpoint, flush=True)
        policy = PeftModel.from_pretrained(
            policy_base, checkpoint / "adapter", is_trainable=True
        )

    policy.enable_input_require_grads()
    policy.train()

    reference = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )
    reference.config.use_cache = False
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEGACY_SETTINGS["learning_rate"])
    if checkpoint is not None:
        names = [name for name, p in policy.named_parameters() if p.requires_grad]
        if names != saved["trainable_names"]:
            raise RuntimeError("Trainable parameter order changed; cannot restore optimizer")
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        del saved
        print("optimizer_and_rng_restored: PASS", flush=True)

    next_index = int(saved_state["next_record_index"])
    losses = list(saved_state.get("losses", []))
    if next_index >= len(rows):
        print("training_already_complete:", next_index, flush=True)
        return

    last_saved_index = next_index
    slowest_step = 0.0

    for index in range(next_index, len(rows)):
        if stop_requested or time.time() + max(120.0, 1.5 * slowest_step) >= deadline:
            break

        step_started = time.monotonic()
        row = rows[index]
        print(
            f"processing index={index} sample_id={row['sample_id']}", flush=True
        )
        prompt, encode = make_inputs(processor, row)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            ref_chosen = response_logp(
                reference, processor, prompt, encode, row["chosen"], False
            )
            ref_rejected = response_logp(
                reference, processor, prompt, encode, row["rejected"], False
            )
        policy_chosen = response_logp(
            policy, processor, prompt, encode, row["chosen"], True
        )
        policy_rejected = response_logp(
            policy, processor, prompt, encode, row["rejected"], True
        )

        margin = LEGACY_SETTINGS["beta"] * (
            (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
        )
        loss = -F.logsigmoid(margin)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at record index {index}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        if norm.item() <= 0:
            raise RuntimeError("No nonzero LoRA gradients")
        if any(parameter.grad is not None for parameter in reference.parameters()):
            raise RuntimeError("Reference received gradients")
        optimizer.step()

        losses.append(float(loss.item()))
        next_index = index + 1
        print(
            f"record={next_index}/{len(rows)} loss={loss.item():.8f} "
            f"grad_norm={norm.item():.6f} "
            f"step_seconds={time.monotonic() - step_started:.1f}",
            flush=True,
        )
        slowest_step = max(slowest_step, time.monotonic() - step_started)
        del loss, policy_chosen, policy_rejected, ref_chosen, ref_rejected, encode, prompt

        if next_index % args.save_every == 0:
            state = {"next_record_index": next_index, "losses": losses, "contract": contract}
            save_checkpoint(policy, optimizer, checkpoint_root, state)
            last_saved_index = next_index

    state = {"next_record_index": next_index, "losses": losses, "contract": contract}
    if next_index != last_saved_index or checkpoint is None:
        save_checkpoint(policy, optimizer, checkpoint_root, state)

    if next_index == len(rows):
        write_summary(args.output, state, len(rows))
        print("VANILLA_DPO_TRAINING: PASS", flush=True)
    else:
        print(
            "TRAINING_PAUSED_FOR_NEXT_JOB:",
            f"next_record_index={next_index}",
            f"elapsed_seconds={time.monotonic() - started:.1f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=2700)
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()
    if args.max_seconds <= 120 or args.save_every < 1:
        parser.error("max-seconds must exceed 120; save-every must be positive")
    with output_lock(args.output):
        train(args)


if __name__ == "__main__":
    from run_unified import legacy_main
    raise SystemExit(legacy_main("train", "qwen25"))
