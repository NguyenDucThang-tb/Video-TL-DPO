"""Exact CPU resume regression; skipped when the optional ML packages are absent."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ML_AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ("torch", "numpy", "safetensors"))


@unittest.skipUnless(ML_AVAILABLE, "requires torch, numpy and safetensors; no GPU needed")
class UnifiedResumeTests(unittest.TestCase):
    def test_accumulated_updates_restore_scheduler_order_optimizer_and_rng(self):
        import random
        import numpy as np
        import torch
        from safetensors.torch import save_file, load_file
        spec = importlib.util.spec_from_file_location("unified_checkpoint_helpers",
            Path(__file__).resolve().parents[1] / "scripts/train_videohallucer_vanilla_resumable.py")
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.drop = torch.nn.Dropout(0.4)
                self.linear = torch.nn.Linear(4, 2)

            def forward(self, x):
                return self.linear(self.drop(x))

            def save_pretrained(self, directory):
                directory.mkdir(parents=True)
                save_file(self.state_dict(), str(directory / "adapter_model.safetensors"))

        def setup():
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            model = Tiny()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 / (s + 1))
            return model, opt, sched

        def updates(model, opt, sched, records):
            losses = []
            for _ in range(records // 8):
                opt.zero_grad(set_to_none=True)
                for _ in range(8):
                    x = torch.randn(2, 4) * (random.random() + np.random.random())
                    logits = model(x)
                    loss = -torch.nn.functional.logsigmoid(logits[:, 0] - logits[:, 1]).mean()
                    losses.append(loss.item())
                    (loss / 8).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                sched.step()
            return losses

        full, fo, fs = setup()
        expected_losses = updates(full, fo, fs, 24)
        first, opt, sched = setup()
        losses = updates(first, opt, sched, 8)
    contract = {"protocol": "unified_adaptive_seed_v2"}
        state = {"next_record_index": 8, "optimizer_steps": 1, "order": list(range(24)),
                 "losses": losses, "contract": contract}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            helpers.save_checkpoint(first, opt, root, state, scheduler=sched)
            checkpoint = helpers.find_checkpoint(root, contract, 24)
            saved = torch.load(checkpoint / "trainer.pt", weights_only=False)
            resumed, opt2, sched2 = setup()
            resumed.load_state_dict(load_file(str(checkpoint / "adapter/adapter_model.safetensors")))
            opt2.load_state_dict(saved["optimizer"])
            sched2.load_state_dict(saved["scheduler"])
            helpers.restore_rng(saved["rng"])
            self.assertEqual(saved["state"]["order"], list(range(24)))
            self.assertEqual(losses + updates(resumed, opt2, sched2, 16), expected_losses)
            self.assertEqual(fs.state_dict(), sched2.state_dict())
            for a, b in zip(full.parameters(), resumed.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
