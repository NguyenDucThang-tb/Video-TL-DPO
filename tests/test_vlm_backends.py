import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from video_tldpo.models.vlm_backends import backend_spec, language_lora_targets, model_identity


def test_backend_model_ids_are_the_confirmed_8b_instruct_checkpoints():
    assert backend_spec("qwen3")["model_id"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert backend_spec("internvl35")["model_id"] == "OpenGVLab/InternVL3_5-8B-Instruct"
    with pytest.raises(ValueError):
        backend_spec("qwen25")


@pytest.mark.parametrize("backend,model_type", [("qwen3", "qwen3_vl"), ("internvl35", "internvl_chat")])
def test_model_identity_checks_config_index_shards_and_backend_code(tmp_path, backend, model_type):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "model_revision.json").write_text(json.dumps({
        "repo_id": backend_spec(backend)["model_id"], "revision": "test-commit"}))
    (tmp_path / "weights.safetensors").write_bytes(b"weights")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "weights.safetensors"}}))
    if backend == "internvl35":
        for name in ("modeling_internvl_chat.py", "modeling_intern_vit.py",
                     "configuration_internvl_chat.py", "configuration_intern_vit.py"):
            (tmp_path / name).write_text("# pinned model source")
    result = model_identity(tmp_path, backend)
    assert result["model_type"] == model_type
    assert result["weight_shards"] == ["weights.safetensors"]
    if backend == "internvl35":
        assert len(result["remote_code"]) == 4


def test_model_identity_refuses_cross_backend_and_missing_remote_code(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_vl"}')
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "model_revision.json").write_text(json.dumps({
        "repo_id": backend_spec("qwen3")["model_id"], "revision": "test-commit"}))
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map":{"x":"w.safetensors"}}')
    (tmp_path / "w.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="model_type"):
        model_identity(tmp_path, "internvl35")
    (tmp_path / "config.json").write_text('{"model_type":"internvl_chat"}')
    with pytest.raises(ValueError, match="remote model code"):
        model_identity(tmp_path, "internvl35")


def test_lora_targets_are_exact_language_linears_only():
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.language_model.layers[0].q_proj = torch.nn.Linear(4, 4)
            self.language_model.layers[0].mlp = torch.nn.Module()
            self.language_model.layers[0].mlp.down_proj = torch.nn.Linear(4, 4)
            self.visual = torch.nn.Module()
            self.visual.q_proj = torch.nn.Linear(4, 4)

    names = language_lora_targets(Tiny(), "qwen3")
    assert names == ["language_model.layers.0.q_proj", "language_model.layers.0.mlp.down_proj"]
    assert all("visual" not in name for name in names)
