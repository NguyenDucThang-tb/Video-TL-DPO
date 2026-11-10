"""Versioned experiment contract from unified_config.md (no GPU imports)."""
import math
import re

PROTOCOL_ID = "unified_adaptive_seed_v2"
PROMPT_SUFFIX = "\nAnswer with yes or no."
SETTINGS = {
    "protocol": PROTOCOL_ID, "frames": 8, "resolution": 448,
    "beta": 0.1, "learning_rate": 1e-4, "weight_decay": 0.0,
    "seed": 0, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "epochs": 1, "batch_size": 1, "grad_accum": 8, "clip_grad_norm": 1.0,
    "scheduler": "cosine", "warmup_fraction": 0.05,
    "response_target": "answer_token_only", "checkpoint_selection": "last_epoch",
    "dataset_revision": "c93475bf1be466c1af90d005749ccb4e6b3ad066",
}
LORA_PATTERNS = {
    "qwen25": r"(.*\.layers\.\d+\.self_attn\.(q|k|v|o)_proj)|(.*visual\.merger\.mlp\.(0|2))",
    "qwen3": r"(.*\.layers\.\d+\.self_attn\.(q|k|v|o)_proj)|(.*visual\.merger\.(mlp\.(0|2)|linear_fc[12]))",
    "internvl35": r"(.*language_model\.model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj)|(.*mlp1\.(1|3))",
}
ANSWER_WORDS = {"qwen25": ("Yes", "No"), "qwen3": ("Yes", "No"),
                "internvl35": ("Yes", "No")}
# VH correct questions, VH correct pairs, EH correct questions.
BASE_COUNTS = {"qwen25": (230, 78, 270), "qwen3": (235, 89, 264),
               "internvl35": (212, 66, 198)}
MODEL_NAMES = {"qwen25": "Qwen2.5-VL-7B-Instruct", "qwen3": "Qwen3-VL-8B-Instruct",
               "internvl35": "InternVL3_5-8B-Instruct"}


def source_fingerprint():
    import hashlib
    from pathlib import Path
    package = Path(__file__).resolve().parent
    root = package.parents[1]
    files = [package / "unified.py"] + [package / name / "unified.py" for name in (
        "data", "models", "training", "evaluation")] + [root / "scripts/run_unified.py",
        root / "scripts/train_videohallucer_vanilla_resumable.py"]
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(root)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def optimizer_budget(records):
    # Same integer rounding and partial final batch as Adaptive-SEED's trainer.
    return math.ceil(records / SETTINGS["grad_accum"]), int(
        SETTINGS["warmup_fraction"] * records / SETTINGS["grad_accum"])


def frame_indices(total):
    """Always eight frames; short clips repeat indices, never reduce the count."""
    if total < 1:
        raise ValueError("Video contains no frames")
    return [round(i * (total - 1) / 7) for i in range(8)]


def lora_target_names(names, backend):
    return [name for name in names if re.fullmatch(LORA_PATTERNS[backend], name)]
