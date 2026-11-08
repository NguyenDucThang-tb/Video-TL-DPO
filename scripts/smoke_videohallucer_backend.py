"""One-record H200 smoke: native video, teacher-forced DPO, LoRA grads, save, greedy decode."""
import argparse
from pathlib import Path
import shutil

import torch
import torch.nn.functional as F

from video_tldpo.models.vlm_backends import (
    generate, language_lora_targets, load_backend, prepare, response_logp,
)
from train_videohallucer_vanilla_resumable import load_rows, SETTINGS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("qwen3", "internvl35"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run this smoke test on an allocated GPU")
    rows = load_rows(a.data)
    row = rows[0]
    processor, base, identity = load_backend(a.model, a.backend, trainable=True)
    from peft import LoraConfig, get_peft_model
    targets = language_lora_targets(base, a.backend)
    adapter_cfg = LoraConfig(r=SETTINGS["lora_r"], lora_alpha=SETTINGS["lora_alpha"],
                             lora_dropout=SETTINGS["lora_dropout"], bias="none",
                             task_type="CAUSAL_LM" if a.backend == "qwen3" else None,
                             target_modules=targets)
    policy = get_peft_model(base, adapter_cfg)
    if hasattr(policy, "enable_input_require_grads"):
        policy.enable_input_require_grads()
    policy.train()
    _, reference, _ = load_backend(a.model, a.backend, trainable=False)
    reference.eval().requires_grad_(False)
    prepared = prepare(processor, row, a.backend, torch.device("cpu"))
    optimizer = torch.optim.AdamW([x for x in policy.parameters() if x.requires_grad], lr=1e-5)
    before = [x.detach().clone() for x in policy.parameters() if x.requires_grad]
    with torch.no_grad():
        ref_c = response_logp(reference, processor, prepared, row["chosen"], False)
        ref_r = response_logp(reference, processor, prepared, row["rejected"], False)
    pi_c = response_logp(policy, processor, prepared, row["chosen"], True)
    pi_r = response_logp(policy, processor, prepared, row["rejected"], True)
    loss = -F.logsigmoid(SETTINGS["beta"] * ((pi_c - ref_c) - (pi_r - ref_r)))
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite DPO loss")
    loss.backward()
    trainable = [x for x in policy.parameters() if x.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
    if norm.item() <= 0 or any(x.grad is not None for x in reference.parameters()):
        raise RuntimeError("LoRA gradient or reference-freeze check failed")
    optimizer.step()
    if not any(not torch.equal(old, new.detach()) for old, new in zip(before, trainable)):
        raise RuntimeError("LoRA parameters did not change after optimizer step")
    if any("language_model." not in name for name, x in policy.named_parameters() if x.requires_grad):
        raise RuntimeError("A visual/projector parameter unexpectedly became trainable")
    a.output.mkdir(parents=True, exist_ok=True)
    adapter_dir = a.output / "adapter"
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    policy.save_pretrained(adapter_dir)
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise RuntimeError("Adapter save failed")
    policy.eval()
    answer, token_count = generate(policy, processor, prepared, a.backend, 16)
    if not answer.strip() or token_count < 1:
        raise RuntimeError("Greedy answer generation returned empty output")
    print("backend:", a.backend)
    print("model_identity:", identity)
    print("sample_id:", row["sample_id"], "frames:", len(prepared.frame_indices))
    print("policy_chosen:", float(pi_c.detach()), "policy_rejected:", float(pi_r.detach()))
    print("reference_chosen:", float(ref_c), "reference_rejected:", float(ref_r))
    print("dpo_loss:", float(loss.detach()), "grad_norm:", float(norm))
    print("generated_answer:", repr(answer), "generated_tokens:", token_count)
    print("adapter:", adapter_dir)
    print("NATIVE_VIDEO_DPO_SMOKE_{}: PASS".format(a.backend.upper()))


if __name__ == "__main__":
    main()
