"""GPU diagnostic for the unified I/O and LoRA scope; not a training result."""
import argparse
import os
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from video_tldpo.models.unified import load, prepare, answer_ids, answer_scores, attach_lora
from video_tldpo.unified import LORA_SCOPES, SETTINGS, validate_language_attention_parameters


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=('qwen25', 'qwen3', 'internvl35'), required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lora-scope', choices=LORA_SCOPES,
                   default=os.environ.get('VIDEO_TLDPO_LORA_SCOPE', 'attention_projector'))
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Run in an allocated GPU job')
    adapter = a.output / 'adapter'
    if adapter.exists():
        raise ValueError('Smoke adapter already exists; choose a fresh output directory')
    row = json.loads(a.data.read_text().splitlines()[0])
    processor, model, tokenizer = load(a.model, a.backend, trainable=True)
    policy, targets = attach_lora(model, a.backend, a.lora_scope)
    if a.lora_scope == 'language_attention':
        validate_language_attention_parameters(
            [name for name, parameter in policy.named_parameters() if parameter.requires_grad], a.backend)
        print('language_attention_trainables: PASS', flush=True)
    _, reference, _ = load(a.model, a.backend)
    reference.eval().requires_grad_(False)
    policy.train()
    batch, indices = prepare(processor, policy, row, a.backend)
    tokens = answer_ids(tokenizer, a.backend)
    with torch.no_grad():
        ref, _ = answer_scores(reference, batch, tokens)
    scores, _ = answer_scores(policy, batch, tokens)
    chosen = 0 if row['gt_answer'].lower() == 'yes' else 1
    rejected = 1 - chosen
    loss = -F.logsigmoid(SETTINGS['beta'] * ((scores[chosen] - ref[chosen]) - (scores[rejected] - ref[rejected])))
    trainable = [x for x in policy.parameters() if x.requires_grad]
    before = [x.detach().clone() for x in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=SETTINGS['learning_rate'], weight_decay=0.0)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
    if norm.item() <= 0 or any(x.grad is not None for x in reference.parameters()):
        raise RuntimeError('LoRA/reference gradient check failed')
    optimizer.step()
    if not any(not torch.equal(old, new) for old, new in zip(before, trainable)):
        raise RuntimeError('LoRA parameters did not update')
    adapter.mkdir(parents=True)
    policy.save_pretrained(adapter)
    policy.eval()
    with torch.inference_mode():
        _, margin = answer_scores(policy, batch, tokens)
    print('lora_targets:', targets)
    print('lora_scope:', a.lora_scope, 'trainable_parameters:', sum(p.numel() for p in trainable))
    print('sample_id:', row['sample_id'], 'frame_indices:', indices, 'answer_tokens:', tokens)
    print('loss:', loss.item(), 'grad_norm:', norm.item(), 'margin:', margin.item())
    print('UNIFIED_GPU_DIAGNOSTIC: PASS; full budget/base reproduction NOT VERIFIED')


if __name__ == '__main__':
    main()
