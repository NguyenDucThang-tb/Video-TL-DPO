"""Lazy Qwen2.5-VL backend; import module không yêu cầu GPU/model cache."""
def load_qwen(model_id="Qwen/Qwen2.5-VL-7B-Instruct", dtype=None, device_map="auto"):
    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    except ImportError as e: raise RuntimeError("Cài torch, transformers trong GPU environment") from e
    dtype = dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, device_map=device_map)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor
