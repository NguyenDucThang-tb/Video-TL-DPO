"""Vanilla DPO loss."""
import torch.nn.functional as F

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    loss = -F.logsigmoid(margin)
    return loss.mean(), {"margin": margin.detach(), "accuracy": (margin > 0).float().mean().detach()}
