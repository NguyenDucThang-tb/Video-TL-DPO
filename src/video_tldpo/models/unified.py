"""The same prompt, frame geometry and answer-token scoring for all three models."""
from pathlib import Path

from video_tldpo.unified import ANSWER_WORDS, LORA_PATTERNS, PROMPT_SUFFIX


def identity(model_dir, backend):
    import json
    from video_tldpo.evaluation.video_validation import file_sha256
    root = Path(model_dir)
    expected = {"qwen25": "qwen2_5_vl", "qwen3": "qwen3_vl", "internvl35": "internvl_chat"}[backend]
    if json.loads((root / "config.json").read_text()).get("model_type") != expected:
        raise ValueError("Unexpected model type for " + backend)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(not (root / name).is_file() for name in shards):
        raise ValueError("Missing model shards")
    files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]
    if backend != "internvl35":
        files.append("preprocessor_config.json")
    if backend != "qwen25":
        manifest = json.loads((root / "model_revision.json").read_text())
        expected_repo = "Qwen/Qwen3-VL-8B-Instruct" if backend == "qwen3" else "OpenGVLab/InternVL3_5-8B-Instruct"
        if manifest.get("repo_id") != expected_repo or not manifest.get("revision"):
            raise ValueError("Model revision manifest differs from selected backbone")
        files.append("model_revision.json")
    if backend == "internvl35":
        files.extend(["modeling_internvl_chat.py", "modeling_intern_vit.py",
                      "configuration_internvl_chat.py", "configuration_intern_vit.py"])
    for name in ("tokenizer.json", "video_preprocessor_config.json", "chat_template.json",
                 "chat_template.jinja", "special_tokens_map.json", "generation_config.json"):
        if (root / name).is_file():
            files.append(name)
    return {"backend": backend, "files": {name: file_sha256(root / name) for name in files},
            "weight_shards": {name: (root / name).stat().st_size for name in shards}}


def load(model_dir, backend, trainable=False):
    import torch
    from transformers import AutoProcessor
    if backend in ("qwen25", "qwen3"):
        import transformers
        cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration" if backend == "qwen25"
                      else "Qwen3VLForConditionalGeneration")
        processor = AutoProcessor.from_pretrained(model_dir, use_fast=False, local_files_only=True)
        model = cls.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map={"": 0},
                                    attn_implementation="sdpa", local_files_only=True)
        tokenizer = processor.tokenizer
    else:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        config.llm_config._attn_implementation = "eager"
        model = AutoModel.from_pretrained(model_dir, config=config, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, use_flash_attn=False, trust_remote_code=True,
            local_files_only=True).to("cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True,
                                                 use_fast=False, local_files_only=True)
        model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if model.img_context_token_id is None or model.img_context_token_id == tokenizer.unk_token_id:
            raise ValueError("Missing InternVL image context token")
        if model.num_image_token != 256:
            raise ValueError("Unified protocol requires 256 image tokens/frame")
        processor = tokenizer
    model.requires_grad_(False)
    model.config.use_cache = False
    if backend == "internvl35":
        model.language_model.config.use_cache = False
    if trainable:
        candidate = model if backend != "internvl35" else model.language_model
        candidate.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        candidate.enable_input_require_grads()
    model.eval()
    return processor, model, tokenizer


def decode(path):
    from decord import VideoReader, cpu
    import numpy as np
    import math
    from PIL import Image
    reader = VideoReader(path, ctx=cpu(0))
    total = len(reader)
    if total < 1:
        raise ValueError("Video contains no frames: " + path)
    fps = float(reader.get_avg_fps() or 30.0)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid video FPS: " + path)
    indices = np.linspace(0, total - 1, 8).round().astype(int).tolist()
    frames = [Image.fromarray(a).convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
              for a in reader.get_batch(indices).asnumpy()]
    duration = total / fps
    return frames, 8 / duration, indices


def prepare(processor, model, row, backend):
    import torch
    import numpy as np
    frames, fps, indices = decode(row["video_path"])
    question = row["question"].strip() + PROMPT_SUFFIX
    if backend in ("qwen25", "qwen3"):
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        video = torch.from_numpy(np.stack([np.asarray(f) for f in frames])).permute(0, 3, 1, 2)
        kwargs = {"fps": [fps]} if backend == "qwen25" else {
            "video_metadata": [{"total_num_frames": 8, "fps": fps,
                                "duration": 8 / fps, "frames_indices": list(range(8))}],
            "do_sample_frames": False}
        batch = processor(text=[prompt], videos=[video], return_tensors="pt", **kwargs)
    else:
        import copy
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        transform = transforms.Compose([
            transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        template = copy.deepcopy(model.conv_template)
        template.system_message = model.system_message
        prefix = "".join("Frame{}: <image>\n".format(i + 1) for i in range(8))
        template.append_message(template.roles[0], prefix + question)
        template.append_message(template.roles[1], None)
        prompt = template.get_prompt()
        replacement = "<img>" + "<IMG_CONTEXT>" * 256 + "</img>"
        for _ in range(8):
            prompt = prompt.replace("<image>", replacement, 1)
        batch = dict(processor(prompt, return_tensors="pt"))
        if (batch["input_ids"] == model.img_context_token_id).sum().item() != 8 * 256:
            raise ValueError("InternVL visual token count changed")
        batch.update(pixel_values=torch.stack([transform(f) for f in frames]).to(torch.bfloat16),
                     image_flags=torch.ones((8, 1), dtype=torch.long))
    device = next(model.parameters()).device
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}, indices


def answer_ids(tokenizer, backend):
    ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in ANSWER_WORDS[backend]]
    if ids[0] == ids[1]:
        raise ValueError("YES and NO tokens must differ")
    return ids


def answer_scores(model, batch, token_ids):
    """Scores at the last prompt position, with no response/EOS appended."""
    import torch.nn.functional as F
    logits = model(**batch, use_cache=False).logits[:, -1, :].float()
    scores = F.log_softmax(logits, dim=-1)[0, token_ids]
    return scores, (logits[0, token_ids[0]] - logits[0, token_ids[1]])


def attach_lora(model, backend):
    import re
    import torch
    from peft import LoraConfig, get_peft_model
    from video_tldpo.unified import SETTINGS
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and re.fullmatch(LORA_PATTERNS[backend], name)]
    if not any("self_attn" in name for name in targets) or not any(
            "merger" in name or "mlp1." in name for name in targets):
        raise ValueError("Unified LoRA must cover LM attention AND visual projector; found " + repr(targets))
    config = LoraConfig(r=SETTINGS["lora_r"], lora_alpha=SETTINGS["lora_alpha"],
        lora_dropout=SETTINGS["lora_dropout"], bias="none", target_modules=LORA_PATTERNS[backend],
        task_type=None if backend == "internvl35" else "CAUSAL_LM")
    policy = get_peft_model(model, config)
    expected = {"qwen25": 10395648, "internvl35": 15597568}.get(backend)
    actual = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if expected is not None and actual != expected:
        raise ValueError("Unified LoRA parameter budget differs: expected {}, found {}".format(expected, actual))
    return policy, targets
