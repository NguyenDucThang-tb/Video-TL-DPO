"""Backend-specific native video I/O for the additional 8B DPO arms.

The module deliberately imports model libraries lazily so split/preflight tools
remain usable with the system Python on Gadi's login nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re


BACKENDS = {
    "qwen3": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "model_type": "qwen3_vl",
        "protocol": "qwen3_native_video_fixed8_v1",
    },
    "internvl35": {
        "model_id": "OpenGVLab/InternVL3_5-8B-Instruct",
        "model_type": "internvl_chat",
        "protocol": "internvl35_video_frames_448_fixed8_v1",
    },
}
TARGET_LEAVES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def backend_spec(name: str) -> dict:
    try:
        return BACKENDS[name]
    except KeyError as exc:
        raise ValueError("backend must be qwen3 or internvl35") from exc


def model_identity(model_dir: Path, backend: str) -> dict:
    """Cheap local identity checks; full weight inventory is captured at staging."""
    spec = backend_spec(backend)
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_type") != spec["model_type"]:
        raise ValueError("{} expected model_type {!r}, found {!r}".format(
            backend, spec["model_type"], config.get("model_type")))
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards or any(not (model_dir / item).is_file() for item in shards):
        raise ValueError("Model safetensors index is empty or has missing shards")
    if backend == "internvl35":
        for name in ("modeling_internvl_chat.py", "modeling_intern_vit.py",
                     "configuration_internvl_chat.py", "configuration_intern_vit.py"):
            if not (model_dir / name).is_file():
                raise ValueError("InternVL remote model code is missing: " + name)
    revision_path = model_dir / "model_revision.json"
    if not revision_path.is_file():
        raise ValueError("Pinned model revision manifest is missing: " + str(revision_path))
    revision = json.loads(revision_path.read_text())
    if revision.get("repo_id") != spec["model_id"] or not revision.get("revision"):
        raise ValueError("Pinned model revision manifest does not match selected backend")
    return {"backend": backend, "model_id": spec["model_id"], "model_type": config["model_type"],
            "revision": revision["revision"], "revision_manifest_sha256": _sha(revision_path),
            "config_sha256": _sha(model_dir / "config.json"),
            "index_sha256": _sha(index_path),
            "tokenizer_config_sha256": _sha(model_dir / "tokenizer_config.json"),
            "weight_shards": shards,
            "remote_code": {n: _sha(model_dir / n) for n in (
                "modeling_internvl_chat.py", "modeling_intern_vit.py",
                "configuration_internvl_chat.py", "configuration_intern_vit.py")
                if (model_dir / n).is_file()}}


def _sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class PreparedVideo:
    prompt: str
    encode: object
    frame_indices: list[int]
    source_fps: float
    source_frame_count: int


def _sample_video(path: str, maximum: int = 8):
    from decord import VideoReader, cpu
    import numpy as np
    reader = VideoReader(path, ctx=cpu(0))
    total = len(reader)
    if total < 1:
        raise ValueError("Video contains no decodable frames: " + path)
    fps = float(reader.get_avg_fps() or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video has invalid source FPS: " + path)
    count = min(maximum, total)
    if count > 1 and count % 2:
        count -= 1
    count = max(2, count)
    indices = np.linspace(0, total - 1, count).round().astype(int).tolist() if total > 1 else [0, 0]
    frames = reader.get_batch(indices).asnumpy()
    return frames, indices, fps, total


def prepare(processor, row: dict, backend: str, device):
    backend_spec(backend)
    frames, indices, fps, total = _sample_video(row["video_path"])
    question = row["question"].strip() + " Answer yes or no."
    if backend == "qwen3":
        return _prepare_qwen3(processor, frames, indices, fps, total, question, device)
    return _prepare_internvl(processor, frames, indices, fps, total, question, device)


def _prepare_qwen3(processor, frames, indices, fps, total, question, device):
    import torch
    import numpy as np
    from transformers.video_utils import VideoMetadata
    frame_tensor = torch.from_numpy(np.asarray(frames)).permute(0, 3, 1, 2)
    duration = total / fps
    metadata = VideoMetadata(total_num_frames=total, fps=fps, duration=duration)
    metadata.frames_indices = indices
    messages = [{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    cache = {prompt: processor(text=[prompt], videos=[frame_tensor], video_metadata=[metadata],
                               do_sample_frames=False, return_tensors="pt")}
    prompt_ids = cache[prompt]["input_ids"][0].tolist()

    def encode(text, target_device):
        if text not in cache:
            batch = processor(text=[text], videos=[frame_tensor], video_metadata=[metadata],
                              do_sample_frames=False, return_tensors="pt")
            if batch["input_ids"][0, :len(prompt_ids)].tolist() != prompt_ids:
                raise RuntimeError("Qwen3 prompt/response token prefix changed")
            cache[text] = batch
        result = {k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in cache[text].items()}
        return result

    # A one-frame clip is retained if the decoder exposes only one frame. Record
    # the actual sampled source indices/FPS for experiment provenance.
    return PreparedVideo(prompt, encode, indices, fps, total)


def _prepare_internvl(processor, frames, indices, fps, total, question, device):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    model = processor
    tokenizer = model
    # The model tokenizer/template are attached after loading by load_backend.
    tokenizer = getattr(processor, "tokenizer", tokenizer)
    image_transform = transforms.Compose([
        transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    rgb_frames = [Image.fromarray(np.asarray(frame)).convert("RGB") for frame in frames]
    frame_prompt = "".join("Frame{}: <image>\n".format(i + 1) for i in range(len(rgb_frames)))
    template = model.conv_template.copy() if hasattr(model.conv_template, "copy") else __import__("copy").deepcopy(model.conv_template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], frame_prompt + question)
    template.append_message(template.roles[1], None)
    prompt = template.get_prompt()
    image_context = "<img>" + "<IMG_CONTEXT>" * model.num_image_token + "</img>"
    for _ in rgb_frames:
        prompt = prompt.replace("<image>", image_context, 1)
    model.img_context_token_id = _internvl_context_token_id(tokenizer)
    pixels = torch.stack([image_transform(frame) for frame in rgb_frames]).to(torch.bfloat16)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    visual_tokens = sum(token_id == model.img_context_token_id for token_id in prompt_ids)
    expected_visual_tokens = len(rgb_frames) * int(model.num_image_token)
    if visual_tokens != expected_visual_tokens:
        raise RuntimeError("InternVL prompt has {} visual context tokens; expected {}".format(
            visual_tokens, expected_visual_tokens))
    cache = {}

    def encode(text, target_device):
        if text not in cache:
            tokens = tokenizer(text, return_tensors="pt")
            ids = tokens["input_ids"]
            if ids[0, :len(prompt_ids)].tolist() != prompt_ids:
                raise RuntimeError("InternVL prompt/response token prefix changed")
            cache[text] = {"input_ids": ids, "attention_mask": tokens["attention_mask"],
                           "pixel_values": pixels,
                           "image_flags": torch.ones((len(rgb_frames), 1), dtype=torch.long)}
        return {k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in cache[text].items()}

    return PreparedVideo(prompt, encode, indices, fps, total)


def _internvl_context_token_id(tokenizer) -> int:
    token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise RuntimeError("InternVL tokenizer does not recognize <IMG_CONTEXT>")
    return int(token_id)


def load_backend(model_dir: Path, backend: str, trainable: bool = False):
    import torch
    model_dir = Path(model_dir)
    identity = model_identity(model_dir, backend)
    if backend == "qwen3":
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True)
    else:
        from transformers import AutoModel, AutoTokenizer
        processor = AutoModel.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True, local_files_only=True).to("cuda:0")
        processor.eval()
        processor.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, use_fast=False, local_files_only=True)
        # Each independently loaded model (policy and frozen reference) needs
        # this ID before forward(), which uses it to replace image token embeds.
        processor.img_context_token_id = _internvl_context_token_id(processor.tokenizer)
        model = processor
    for candidate in (getattr(model, "config", None),
                      getattr(model, "language_model", None) and model.language_model.config,
                      getattr(getattr(model, "language_model", None), "model", None)
                      and model.language_model.model.config):
        if candidate is not None and hasattr(candidate, "use_cache"):
            candidate.use_cache = False
    if trainable:
        _enable_gradient_checkpointing(model, backend)
    model.train(trainable)
    return processor, model, identity


def _enable_gradient_checkpointing(model, backend):
    candidates = [model]
    if backend == "internvl35":
        candidates.extend([getattr(model, "language_model", None),
                           getattr(getattr(model, "language_model", None), "model", None)])
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "gradient_checkpointing_enable"):
            candidate.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            return
    raise RuntimeError("Backend exposes no compatible gradient checkpointing API")


def language_lora_targets(model, backend: str) -> list[str]:
    """Return exact language-only projection paths; never suffix-match vision layers."""
    import torch
    if backend == "qwen3":
        prefix = "language_model."
    elif backend == "internvl35":
        prefix = "language_model."
    else:
        backend_spec(backend)
    found = [name for name, module in model.named_modules()
             if "language_model." in name and name.rsplit(".", 1)[-1] in TARGET_LEAVES
             and isinstance(module, torch.nn.Linear)]
    if not found:
        raise RuntimeError("No language-only LoRA projections found for " + backend)
    if any(not re.search(r"(?:^|\.)language_model\.", name) for name in found):
        raise RuntimeError("LoRA target escaped language_model")
    return found


def response_logp(model, processor, prepared: PreparedVideo, response: str, with_grad: bool):
    import torch
    import torch.nn.functional as F
    device = next(model.parameters()).device
    batch = prepared.encode(prepared.prompt + response, device)
    response_ids = processor.tokenizer(response, add_special_tokens=False)["input_ids"] \
        if hasattr(processor, "tokenizer") else processor(response, add_special_tokens=False)["input_ids"]
    # InternVL's tokenizer object is installed on the loaded model; Qwen's processor
    # has a .tokenizer property. Both tokenize the response without BOS/EOS.
    if not response_ids:
        raise RuntimeError("Response tokenized to an empty sequence")
    if batch["input_ids"][0, -len(response_ids):].tolist() != response_ids:
        raise RuntimeError("Response token alignment check failed")
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        logits = model(**batch, use_cache=False).logits
        count = len(response_ids)
        selected = logits[:, -(count + 1):-1, :].float()
        labels = batch["input_ids"][:, -count:]
        return F.log_softmax(selected, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1).sum()


def generate(model, processor, prepared: PreparedVideo, backend: str, max_new_tokens=16):
    import torch
    device = next(model.parameters()).device
    batch = prepared.encode(prepared.prompt, device)
    generation_inputs = dict(batch)
    generation_inputs.pop("image_flags", None)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }
    # InternVL's custom generate() passes use_cache=True to its language model
    # internally; forwarding it here duplicates the keyword.
    if backend == "qwen3":
        generation_kwargs["use_cache"] = True
    with torch.inference_mode():
        output = model.generate(**generation_inputs, **generation_kwargs)
    ids = output[0]
    input_ids = batch["input_ids"][0]
    if ids.numel() >= input_ids.numel() and torch.equal(ids[:input_ids.numel()], input_ids):
        ids = ids[input_ids.numel():]
    if backend == "qwen3":
        answer = processor.batch_decode(ids.unsqueeze(0), skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)[0]
    else:
        answer = processor.tokenizer.decode(ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
    return answer, int(ids.numel())
