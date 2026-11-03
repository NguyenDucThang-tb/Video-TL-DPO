"""Entry point có guard GPU; chưa tự động submit hoặc chạy model trên login node."""
import argparse
from pathlib import Path
from video_tldpo.config import Config

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--smoke", action="store_true"); a=p.parse_args()
    c=Config.from_yaml(a.config)
    try:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("NOT RUN: CUDA GPU không khả dụng; chạy qua PBS trên Gadi")
    except ImportError: raise RuntimeError("NOT RUN: chưa cài torch GPU")
    print(f"READY: {c.mode}; model={c.model_id}; frames={c.frames}; output={c.output_root}")
    print("TODO_RUNTIME: cần model cache và dataset mount để thực hiện forward Qwen2.5-VL.")

if __name__ == "__main__": main()
