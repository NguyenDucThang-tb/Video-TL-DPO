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


def submit_fixture(tmp_path):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "pbs").mkdir()
    (project / "artifacts").mkdir()
    for relative in ("scripts/train_videohallucer_vanilla_resumable.py", "pbs/train_videohallucer_vanilla_resumable.pbs"):
        (project / relative).write_bytes((ROOT / relative).read_bytes())
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"not-a-decoding-test")
    data = project / "artifacts/train_binary.jsonl"
    data.write_text("\n".join(json.dumps({
        "sample_id": str(i), "split": "train", "question": "q", "video_path": str(video),
        "chosen": "Yes.", "rejected": "No.", "gt_answer": "yes",
    }) for i in range(300)) + "\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "weights.safetensors"}}))
    for name in ("weights.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json"):
        (model / name).write_text("fixture")
    archive = tmp_path / "env.tar.gz"
    archive.write_bytes(b"not-an-extraction-test")
    env = dict(os.environ, VIDEO_TLDPO_PROJECT=str(project), VIDEO_TLDPO_MODEL=str(model),
               VIDEO_TLDPO_ARCHIVE=str(archive), VIDEO_TLDPO_DATA=str(data), USER="fixture")
    return project, data, env


def test_submission_preflight_stops_missing_data_before_qsub(tmp_path):
    project, data, env = submit_fixture(tmp_path)
    data.unlink()
    result = subprocess.run(["bash", str(ROOT / "scripts/submit_videohallucer_vanilla_chain.sh")],
                            env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "PREFLIGHT FAILED" in result.stderr
    assert not (project / "logs").exists()


def test_submission_refuses_an_existing_chain(tmp_path):
    project, _, env = submit_fixture(tmp_path)
    bins = tmp_path / "bin"
    bins.mkdir()
    qsub = bins / "qsub"
    qsub.write_text("#!/bin/bash\nexit 99\n")
    qsub.chmod(0o755)
    qselect = bins / "qselect"
    qselect.write_text("#!/bin/bash\nprintf '123.gadi-pbs\\n'\n")
    qselect.chmod(0o755)
    env["PATH"] = str(bins) + os.pathsep + env["PATH"]
    result = subprocess.run(["bash", str(ROOT / "scripts/submit_videohallucer_vanilla_chain.sh")],
                            env=env, text=True, capture_output=True)
    assert result.returncode != 0 and "refusing a duplicate chain" in result.stderr
    assert not (project / "logs").exists()


def test_submission_chains_real_script_arguments_without_arrays(tmp_path):
    project, _, env = submit_fixture(tmp_path)
    bins = tmp_path / "bin"
    bins.mkdir()
    qsub = bins / "qsub"
    qsub.write_text("#!/bin/bash\nset -eu\nn=0\nif [ -f \"$FAKE_COUNTER\" ]; then read -r n < \"$FAKE_COUNTER\"; fi\nn=$((n+1))\nprintf '%s\\n' \"$n\" > \"$FAKE_COUNTER\"\nprintf '%s\\n' \"$*\" >> \"$FAKE_ARGS\"\nprintf '%s.gadi-pbs\\n' \"$n\"\n")
    qsub.chmod(0o755)
    qselect = bins / "qselect"
    qselect.write_text("#!/bin/bash\nexit 0\n")
    qselect.chmod(0o755)
    env.update(PATH=str(bins) + os.pathsep + env["PATH"],
               FAKE_COUNTER=str(tmp_path / "counter"), FAKE_ARGS=str(tmp_path / "args"))
    result = subprocess.run(["bash", str(ROOT / "scripts/submit_videohallucer_vanilla_chain.sh"), "--jobs", "3"],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "args").read_text().splitlines()
    assert len(args) == 3 and all("-J" not in line for line in args)
    assert "depend=" not in args[0]
    assert "depend=afterok:1.gadi-pbs" in args[1]
    assert "depend=afterok:2.gadi-pbs" in args[2]
    assert all(str(project / "pbs/train_videohallucer_vanilla_resumable.pbs") in line for line in args)
