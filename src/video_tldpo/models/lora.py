"""Shared LoRA scope for the unified student budget."""
def apply_lora(model, r=16, alpha=32, dropout=0.05, backend="qwen25"):
    from video_tldpo.models.unified import attach_lora
    if (r, alpha, dropout) != (16, 32, 0.05):
        raise ValueError("Unified LoRA budget requires r16/alpha32/dropout0.05")
    return attach_lora(model, backend)[0]
