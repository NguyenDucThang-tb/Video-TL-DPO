from .vanilla_dpo import dpo_loss
from .target_generation import target_generation_loss
from .target_condition import target_condition_loss

def combined_loss(mode, scores, beta=0.1, lambda_condition=1.0):
    if mode == "vanilla_dpo":
        return dpo_loss(scores.pc, scores.pr, scores.rc, scores.rr, beta)
    if mode == "target_generation_only":
        return target_generation_loss(scores.pc_tok, scores.pr_tok, scores.rc_tok, scores.rr_tok, scores.mc, scores.mr, beta)
    if mode in {"target_condition_only", "full_tl_dpo", "full_tl_dpo_random_mask"}:
        cond = target_condition_loss(scores.pc_orig, scores.pc_mask, scores.rc_orig, scores.rc_mask, beta)
        if mode == "target_condition_only": return cond
        gen = target_generation_loss(scores.pc_tok, scores.pr_tok, scores.rc_tok, scores.rr_tok, scores.mc, scores.mr, beta)
        return gen[0] + lambda_condition * cond[0], {"generation": gen[1], "condition": cond[1]}
    raise ValueError(mode)
