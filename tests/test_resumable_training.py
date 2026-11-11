"""CPU regressions for resume/scoring/submission; no Qwen weights or real video."""
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.torch import load_file, save_file
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "resumable", ROOT / "scripts/train_videohallucer_vanilla_resumable.py"
)
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


class SmallPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.4)
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(self.dropout(x))

    def save_pretrained(self, directory):
        directory.mkdir(parents=True)
        save_file(self.state_dict(), str(directory / "adapter_model.safetensors"))


def seed():
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)


def updates(policy, optimizer, n):
    losses = []
    for _ in range(n):
        x = torch.randn(2, 4) * (0.5 + random.random() + np.random.random())
        optimizer.zero_grad(set_to_none=True)
        loss = policy(x).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def test_response_logp_uses_causal_shift_float32_and_masks_prompt():
    logits = torch.randn(1, 6, 9, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    actual = resume.response_suffix_logp(logits, ids, 2)
    expected = logits.float().log_softmax(-1)[0, 3, 4] + logits.float().log_softmax(-1)[0, 4, 5]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.count_nonzero(logits.grad[:, :3]) == 0
    assert torch.count_nonzero(logits.grad[:, 3:5]) > 0
    assert torch.count_nonzero(logits.grad[:, 5:]) == 0


def test_resume_matches_uninterrupted_dropout_and_adamw(tmp_path):
    seed()
    full = SmallPolicy()
    full_optimizer = torch.optim.AdamW(full.parameters(), lr=0.01)
    full_losses = updates(full, full_optimizer, 6)

    seed()
    first = SmallPolicy()
    first_optimizer = torch.optim.AdamW(first.parameters(), lr=0.01)
    first_losses = updates(first, first_optimizer, 3)
    contract = {"schema": 2, "data": "unchanged"}
    state = {"next_record_index": 3, "losses": first_losses, "contract": contract}
    resume.save_checkpoint(first, first_optimizer, tmp_path, state)
    checkpoint = resume.find_checkpoint(tmp_path, contract, 6)
    saved = torch.load(checkpoint / "trainer.pt", map_location="cpu", weights_only=False)

    second = SmallPolicy()  # model construction consumes RNG before restoration
    second.load_state_dict(load_file(str(checkpoint / "adapter/adapter_model.safetensors")))
    second_optimizer = torch.optim.AdamW(second.parameters(), lr=0.01)
    second_optimizer.load_state_dict(saved["optimizer"])
    assert saved["trainable_names"] == [name for name, _ in second.named_parameters()]
    resume.restore_rng(saved["rng"])
    second_losses = updates(second, second_optimizer, 3)
    assert first_losses + second_losses == full_losses
    for expected, actual in zip(full.parameters(), second.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for expected, actual in zip(full_optimizer.state.values(), second_optimizer.state.values()):
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)


def make_checkpoint(root, cursor, contract):
    model = SmallPolicy()
    optimizer = torch.optim.AdamW(model.parameters())
    resume.save_checkpoint(model, optimizer, root, {
        "next_record_index": cursor, "losses": [0.5] * cursor, "contract": contract,
    })


def test_resume_refuses_changed_data_and_legacy_checkpoint(tmp_path):
    make_checkpoint(tmp_path, 1, {"data": "original"})
    with pytest.raises(RuntimeError, match="differ"):
        resume.find_checkpoint(tmp_path, {"data": "changed"}, 3)
    (tmp_path / "latest/state.json").unlink()
    with pytest.raises(RuntimeError, match="Legacy"):
        resume.find_checkpoint(tmp_path, {"data": "original"}, 3)


def test_interrupted_promotion_recovers_completed_temporary_checkpoint(tmp_path):
    contract = {"data": "same"}
    make_checkpoint(tmp_path, 1, contract)
    make_checkpoint(tmp_path, 2, contract)
    os.replace(tmp_path / "latest", tmp_path / "checkpoint.tmp")
    assert resume.find_checkpoint(tmp_path, contract, 3).name == "checkpoint.tmp"
    promoted = resume.promote_temporary(tmp_path)
    assert promoted.name == "latest" and not (tmp_path / "checkpoint.tmp").exists()
    assert resume.find_checkpoint(tmp_path, contract, 3) == promoted


def test_incomplete_latest_falls_back_to_previous(tmp_path):
    contract = {"data": "same"}
    make_checkpoint(tmp_path, 1, contract)
    make_checkpoint(tmp_path, 2, contract)
    (tmp_path / "latest/adapter/adapter_model.safetensors").unlink()
    assert resume.find_checkpoint(tmp_path, contract, 3).name == "previous"


def test_corrupt_metadata_falls_back_and_summary_points_to_recovered_adapter(tmp_path):
    contract = {"data": "same"}
    make_checkpoint(tmp_path, 1, contract)
    make_checkpoint(tmp_path, 2, contract)
    (tmp_path / "latest/state.json").write_text("{")
    recovered = resume.find_checkpoint(tmp_path, contract, 2)
    assert recovered.name == "previous"
    state = {"next_record_index": 1, "losses": [0.5], "contract": contract}
    resume.write_summary(tmp_path, state, 1, recovered)
    assert json.loads((tmp_path / "train_summary.json").read_text())["adapter_checkpoint"] == str(recovered / "adapter")


def test_output_lock_rejects_duplicate_trainer(tmp_path):
    with resume.output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another trainer"):
            with resume.output_lock(tmp_path):
                pytest.fail("duplicate lock was acquired")


def test_video_sampled_fps_and_prompt_boundary(monkeypatch):
    seen = []

    def process_vision(messages, return_video_kwargs):
        assert return_video_kwargs is True
        request = messages[0]["content"][0]
        assert "nframes" not in request
        assert request["min_frames"] == request["max_frames"] == 8
        return None, [torch.zeros(8, 3, 2, 2)], {"fps": [0.25]}

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, text, fps, **kwargs):
            seen.append(fps)
            ids = [1, 2] if text == ["prompt"] else [1, 2, 3]
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids)),
                    "pixel_values_videos": torch.zeros(4, 3), "video_grid_thw": torch.tensor([[4, 1, 1]]),
                    "second_per_grid_ts": [2 / fps[0]]}

    monkeypatch.setitem(sys.modules, "qwen_vl_utils", SimpleNamespace(process_vision_info=process_vision))
    prompt, encode = resume.make_inputs(Processor(), {"video_path": "fixture.mp4", "question": "q"})
    batch = encode(prompt + "Yes.", torch.device("cpu"))
    assert seen == [[0.25], [0.25]]
    assert batch["second_per_grid_ts"] == [8.0]


@pytest.mark.parametrize("total_frames, expected", [(2, 2), (3, 2), (5, 4), (6, 6), (7, 6), (8, 8), (120, 8)])
def test_short_video_sampling_with_real_qwen_utils(total_frames, expected):
    vision = pytest.importorskip("qwen_vl_utils.vision_process")
    request = resume.video_sampling_spec("fixture.mp4")
    actual = vision.smart_nframes(request, total_frames=total_frames, video_fps=30.0)
    assert actual == expected
    assert actual <= total_frames and actual % vision.FRAME_FACTOR == 0
    if total_frames >= 8:
        assert actual == vision.smart_nframes({"nframes": 8}, total_frames=total_frames, video_fps=30.0)


def test_old_fixed_frame_request_reproduces_reported_error():
    vision = pytest.importorskip("qwen_vl_utils.vision_process")
    with pytest.raises(ValueError, match=r"interval \[2, 6\].*8"):
        vision.smart_nframes({"nframes": 8}, total_frames=6, video_fps=30.0)


def test_legacy_chain_alias_forwards_jobs_and_preflight(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    alias = scripts / "submit_videohallucer_vanilla_chain.sh"
    alias.write_bytes((ROOT / "scripts/submit_videohallucer_vanilla_chain.sh").read_bytes())
    (scripts / "submit_unified.sh").write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    result = subprocess.run(["bash", str(alias), "--jobs", "3", "--preflight-only"],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["train", "qwen25", "--jobs", "3", "--preflight-only"]


def test_default_pbs_uses_packaged_env_and_unified_entrypoint():
    script = (ROOT / "pbs/run_unified.pbs").read_text()
    assert "walltime=00:30:00" in script
    assert "storage=gdata/hn98+scratch/hn98+scratch/jp09" in script
    assert '"$ENV_ROOT/bin/python" "$ENV_ROOT/bin/conda-unpack"' in script
    assert '"$PROJECT/scripts/run_unified.py"' in script
    assert "bin/activate" not in script
