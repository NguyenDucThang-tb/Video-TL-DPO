"""Khung trainer GPU; forward multimodal cụ thể được cung cấp bởi Qwen adapter."""
import math, torch

class PreferenceTrainer:
    def __init__(self, policy, reference, optimizer, scheduler=None, grad_accum=1, clip=1.0):
        self.policy, self.reference, self.optimizer, self.scheduler = policy, reference, optimizer, scheduler
        self.grad_accum, self.clip = grad_accum, clip

    def step(self, loss):
        if not torch.isfinite(loss): raise FloatingPointError(f"loss không hữu hạn: {loss.item()}")
        (loss / self.grad_accum).backward()

    def update(self):
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip)
        self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler: self.scheduler.step()
