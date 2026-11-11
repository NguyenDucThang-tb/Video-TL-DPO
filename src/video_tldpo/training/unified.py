"""One epoch of answer-token DPO with atomic resume at accumulation boundaries."""
import json
import math
import random
import signal
import time

from video_tldpo.evaluation.video_validation import atomic_json
from video_tldpo.unified import SETTINGS, optimizer_budget


def train(args, rows, contract):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import get_cosine_schedule_with_warmup
    import train_videohallucer_vanilla_resumable as checkpoints
    from video_tldpo.models.unified import answer_ids, answer_scores, attach_lora, load, prepare
    stopped = []
    signal.signal(signal.SIGTERM, lambda *_: stopped.append(True))
    deadline = min(time.time() + args.max_seconds, args.deadline_epoch or float("inf"))
    with checkpoints.output_lock(args.output):
        root = args.output / "trainer_state"
        found = checkpoints.find_checkpoint(root, contract, len(rows))
        if found is None and (args.output / "train_summary.json").exists():
            raise ValueError("Summary exists without matching unified checkpoint; use a new output")
        if found and found.name == "checkpoint.tmp":
            found = checkpoints.promote_temporary(root)
        saved = torch.load(found / "trainer.pt", map_location="cpu", weights_only=False) if found else None
        order = [r["sample_id"] for r in rows]
        random.Random(0).shuffle(order)
        state = saved["state"] if saved else {
            "next_record_index": 0, "optimizer_steps": 0, "losses": [],
            "order": order, "contract": contract}
        if state.get("order") != order:
            raise ValueError("Training sample order differs across resume")
        if state["next_record_index"] % 8 and state["next_record_index"] != len(rows):
            raise ValueError("Checkpoint is not at an optimizer/accumulation boundary")
        if state["optimizer_steps"] != math.ceil(state["next_record_index"] / 8):
            raise ValueError("Checkpoint optimizer step counter differs from record cursor")
        if state["next_record_index"] == len(rows):
            write_summary(args.output, state, found)
            print("training_already_complete:", len(rows), flush=True)
            return 0
        if not torch.cuda.is_available():
            raise RuntimeError("Training requires a CUDA GPU job")
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        processor, base, tokenizer = load(args.model, args.backend, trainable=True)
        if found:
            policy = PeftModel.from_pretrained(base, found / "adapter", is_trainable=True)
        else:
            policy, targets = attach_lora(base, args.backend, args.lora_scope)
            print("lora_targets:", targets, flush=True)
        print("lora_scope:", args.lora_scope, flush=True)
        if args.lora_scope == "language_attention":
            from video_tldpo.unified import lora_target_pattern, validate_language_attention_parameters
            # A resumed adapter must also retain the exact attention-only target regex.
            if policy.peft_config["default"].target_modules != lora_target_pattern(args.backend, args.lora_scope):
                raise ValueError("Attention-only adapter target configuration differs")
            validate_language_attention_parameters(
                [name for name, parameter in policy.named_parameters() if parameter.requires_grad], args.backend)
            print("language_attention_trainables: PASS", flush=True)
        policy.train()
        _, reference, _ = load(args.model, args.backend)
        reference.eval()
        reference.requires_grad_(False)
        tokens = answer_ids(tokenizer, args.backend)
        parameters = [p for p in policy.parameters() if p.requires_grad]
        names = [n for n, p in policy.named_parameters() if p.requires_grad]
        print("trainable_parameters:", sum(p.numel() for p in parameters), flush=True)
        optimizer = torch.optim.AdamW(parameters, lr=SETTINGS["learning_rate"], weight_decay=0.0)
        total_steps, warmup = optimizer_budget(len(rows))
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
        if saved:
            if saved["trainable_names"] != names or saved.get("scheduler") is None:
                raise ValueError("Resume requires matching LoRA names and scheduler state")
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            checkpoints.restore_rng(saved["rng"])
            print("optimizer_scheduler_rng_restored: PASS", flush=True)
            del saved
        by_id = {r["sample_id"]: r for r in rows}
        slowest = 0.0
        while state["next_record_index"] < len(rows):
            if stopped or time.time() + max(120.0, 1.5 * slowest) >= deadline:
                break
            started = time.monotonic()
            start = state["next_record_index"]
            end = min(start + SETTINGS["grad_accum"], len(rows))
            optimizer.zero_grad(set_to_none=True)
            losses = []
            # Finish the group before saving/stopping; no gradients are dropped on resume.
            for cursor in range(start, end):
                row = by_id[order[cursor]]
                batch, _ = prepare(processor, policy, row, args.backend)
                with torch.no_grad():
                    ref, _ = answer_scores(reference, batch, tokens)
                scores, _ = answer_scores(policy, batch, tokens)
                chosen = 0 if row["gt_answer"] == "yes" else 1
                rejected = 1 - chosen
                loss = -F.logsigmoid(SETTINGS["beta"] * (
                    (scores[chosen] - ref[chosen]) - (scores[rejected] - ref[rejected])))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite DPO loss at " + row["sample_id"])
                (loss / SETTINGS["grad_accum"]).backward()
                losses.append(float(loss.item()))
                del batch, ref, scores, loss
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            if any(p.grad is not None for p in reference.parameters()):
                raise RuntimeError("Frozen reference received gradients")
            optimizer.step()
            scheduler.step()
            state["next_record_index"] = end
            state["optimizer_steps"] += 1
            state["losses"].extend(losses)
            torch.cuda.synchronize()
            slowest = max(slowest, time.monotonic() - started)
            print("records={}/{} optimizer_step={}/{} loss={:.6f} grad_norm={:.6f} lr={:.9f}".format(
                end, len(rows), state["optimizer_steps"], total_steps, sum(losses)/len(losses),
                norm.item(), optimizer.param_groups[0]["lr"]), flush=True)
            if state["optimizer_steps"] % args.save_every == 0:
                checkpoints.save_checkpoint(policy, optimizer, root, state, scheduler=scheduler)
        checkpoints.save_checkpoint(policy, optimizer, root, state, scheduler=scheduler)
        if state["next_record_index"] == len(rows):
            write_summary(args.output, state, root / "latest")
            print("UNIFIED_VANILLA_DPO_{}_TRAINING: PASS".format(args.backend.upper()), flush=True)
        else:
            print("TRAINING_PAUSED_FOR_NEXT_JOB:", state["next_record_index"], flush=True)
    return 0


def write_summary(output, state, checkpoint):
    total, warmup = optimizer_budget(len(state["losses"]))
    if state["optimizer_steps"] != total:
        raise ValueError("Completed record cursor has incorrect optimizer budget")
    atomic_json(output / "train_summary.json", {
        "status": "COMPLETED", "records": len(state["losses"]), "contract": state["contract"],
        "optimizer_steps": total, "warmup_steps": warmup,
        "adapter_checkpoint": str(checkpoint / "adapter"),
        "loss_first": state["losses"][0], "loss_last": state["losses"][-1]})
