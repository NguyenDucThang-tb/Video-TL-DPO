"""Video Target Condition Loss.

V và V_masked giữ nguyên câu hỏi/response context; chỉ thay đổi visual evidence.
Điều này là phần mở rộng video của target-condition objective trong TL-DPO.
"""
from .vanilla_dpo import dpo_loss

def target_condition_loss(policy_original, policy_masked, ref_original, ref_masked, beta=0.1):
    """So sánh policy/reference preference cho target response ở V và V_masked."""
    # Delta log-ratio: preference trên video gốc phải mạnh hơn sau khi evidence bị mask.
    po = policy_original - ref_original
    pm = policy_masked - ref_masked
    return dpo_loss(po, pm, po.detach() * 0, pm.detach() * 0, beta)
