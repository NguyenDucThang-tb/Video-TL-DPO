"""Đọc cấu hình YAML và kiểm tra các mode thí nghiệm."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml

MODES = {"vanilla_dpo", "target_generation_only", "target_condition_only", "full_tl_dpo", "full_tl_dpo_random_mask"}

@dataclass
class Config:
    data_root: str = "/scratch/jp09/dd9648/datasets_video_hallu"
    output_root: str = "/g/data/hn98/dd9648/Video-TL-DPO/outputs"
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    frames: int = 8
    resolution: int = 448
    seed: int = 0
    lora_r: int = 16
    lora_alpha: int = 32
    beta: float = 0.1
    learning_rate: float = 1e-4
    epochs: int = 1
    grad_accum: int = 8
    lambda_condition: float = 1.0
    mode: str = "vanilla_dpo"
    target_verified_only: bool = True
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path):
        raw = yaml.safe_load(Path(path).read_text()) or {}
        obj = cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        obj.extra = {k: v for k, v in raw.items() if k not in cls.__dataclass_fields__}
        if obj.mode not in MODES:
            raise ValueError(f"mode không hợp lệ: {obj.mode}; chọn một trong {sorted(MODES)}")
        if obj.frames not in {4, 8, 16, 32}:
            raise ValueError("frames phải là 4, 8, 16 hoặc 32")
        return obj
