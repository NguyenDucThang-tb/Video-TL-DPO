"""Target Generation Loss: DPO chỉ cộng log-probability ở target spans."""
from .vanilla_dpo import dpo_loss
from .log_probs import masked_sum

def target_scores(token_logp, target_mask):
    return masked_sum(token_logp, target_mask)

def target_generation_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected,
                           chosen_mask, rejected_mask, beta=0.1):
    pc, pr = target_scores(policy_chosen, chosen_mask), target_scores(policy_rejected, rejected_mask)
    rc, rr = target_scores(ref_chosen, chosen_mask), target_scores(ref_rejected, rejected_mask)
    loss, info = dpo_loss(pc, pr, rc, rr, beta)
    info.update(target_tokens_chosen=chosen_mask.sum(-1).detach(), target_tokens_rejected=rejected_mask.sum(-1).detach())
    return loss, info
