"""Resumable single-run Vanilla DPO training for VideoHallucer on Gadi.

Each PBS job resumes the same policy, optimizer state, and record cursor from
the shared checkpoint directory, then exits cleanly before its walltime.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 300:
        raise RuntimeError(f"Expected exactly 300 train records, found {len(rows)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("Duplicate sample_id in training JSONL")
    for row in rows:
        for key in ("video_path", "question", "chosen", "rejected"):
            if not row.get(key):
                raise RuntimeError(f"Missing {key} in {row.get('sample_id')}")
        if not Path(row["video_path"]).is_file():
            raise FileNotFoundError(row["video_path"])
    return rows


def make_inputs(processor, row: dict):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": row["video_path"], "nframes": 8},
            {"type": "text", "text": row["question"] + " Answer yes or no."},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    def encode(text: str, device: torch.device):
        batch = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    return prompt, encode


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
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        labels = batch["input_ids"][:, 1:]
        count = len(response_ids)
        return log_probs[:, -count:, :].gather(
            -1, labels[:, -count:].unsqueeze(-1)
        ).squeeze(-1).sum()


def save_checkpoint(policy, optimizer, checkpoint_root: Path, state: dict) -> None:
    """Atomically replace latest while retaining one previous recoverable copy."""
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_root / "checkpoint.tmp"
    latest = checkpoint_root / "latest"
    previous = checkpoint_root / "previous"

    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    policy.save_pretrained(temporary / "adapter")
    torch.save(
        {"optimizer": optimizer.state_dict(), "state": state},
        temporary / "trainer.pt",
    )

    if previous.exists():
        shutil.rmtree(previous)
    if latest.exists():
        os.replace(latest, previous)
    os.replace(temporary, latest)
    print(
        "checkpoint_saved:",
        str(latest),
        "next_record_index:",
        state["next_record_index"],
        flush=True,
    )


def find_checkpoint(checkpoint_root: Path) -> Path | None:
    candidates = [checkpoint_root / "latest", checkpoint_root / "previous"]
    valid = [path for path in candidates if (path / "trainer.pt").is_file()]
    if not valid:
        return None
    return max(
        valid,
        key=lambda path: torch.load(
            path / "trainer.pt", map_location="cpu", weights_only=False
        )["state"]["next_record_index"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=2700)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    rows = load_rows(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.output / "trainer_state"

    print("gpu:", torch.cuda.get_device_name(0), flush=True)
    print("records:", len(rows), flush=True)
    print("output:", args.output, flush=True)

    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, use_fast=False
    )
    policy_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    policy_base.config.use_cache = False
    policy_base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    checkpoint = find_checkpoint(checkpoint_root)
    if checkpoint is None:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        policy = get_peft_model(policy_base, lora_config)
        saved_state = {"next_record_index": 0, "losses": []}
        print("resume_checkpoint: none; starting at record 0", flush=True)
    else:
        print("resume_checkpoint:", checkpoint, flush=True)
        policy = PeftModel.from_pretrained(
            policy_base, checkpoint / "adapter", is_trainable=True
        )
        saved = torch.load(
            checkpoint / "trainer.pt", map_location="cpu", weights_only=False
        )
        saved_state = saved["state"]

    policy.enable_input_require_grads()
    policy.train()

    reference = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)
    if checkpoint is not None:
        optimizer.load_state_dict(saved["optimizer"])

    next_index = int(saved_state["next_record_index"])
    losses = list(saved_state.get("losses", []))
    if next_index >= len(rows):
        print("training_already_complete:", next_index, flush=True)
        return

    started = time.monotonic()
    last_saved_index = next_index

    for index in range(next_index, len(rows)):
        if time.monotonic() - started >= args.max_seconds:
            break

        row = rows[index]
        print(
            f"processing index={index} sample_id={row['sample_id']}", flush=True
        )
        prompt, encode = make_inputs(processor, row)
        optimizer.zero_grad(set_to_none=True)

        policy_chosen = response_logp(
            policy, processor, prompt, encode, row["chosen"], True
        )
        policy_rejected = response_logp(
            policy, processor, prompt, encode, row["rejected"], True
        )
        with torch.no_grad():
            ref_chosen = response_logp(
                reference, processor, prompt, encode, row["chosen"], False
            )
            ref_rejected = response_logp(
                reference, processor, prompt, encode, row["rejected"], False
            )

        margin = 0.1 * (
            (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
        )
        loss = -F.logsigmoid(margin)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at record index {index}")
        loss.backward()
        if not any(parameter.grad is not None for parameter in trainable):
            raise RuntimeError("No LoRA gradients")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        losses.append(float(loss.item()))
        next_index = index + 1
        print(
            f"record={next_index}/{len(rows)} loss={loss.item():.6f} "
            f"gpu_memory_gb={torch.cuda.memory_allocated() / 1024**3:.2f}",
            flush=True,
        )
        del loss
        torch.cuda.empty_cache()

        if next_index % args.save_every == 0:
            state = {"next_record_index": next_index, "losses": losses}
            save_checkpoint(policy, optimizer, checkpoint_root, state)
            last_saved_index = next_index

    state = {"next_record_index": next_index, "losses": losses}
    if next_index != last_saved_index:
        save_checkpoint(policy, optimizer, checkpoint_root, state)

    if next_index == len(rows):
        summary = {
            "status": "COMPLETED",
            "records": len(rows),
            "frames": 8,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_min": min(losses),
            "loss_max": max(losses),
            "adapter_checkpoint": str(checkpoint_root / "latest" / "adapter"),
        }
        (args.output / "train_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        print("VANILLA_DPO_TRAINING: PASS", flush=True)
    else:
        print(
            "TRAINING_PAUSED_FOR_NEXT_JOB:",
            f"next_record_index={next_index}",
            flush=True,
        )


if __name__ == "__main__":
    main()
