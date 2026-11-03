"""Teacher-forced response log-probabilities và mask token."""
import torch

def token_log_probs(logits, labels):
    """Trả về log p(y_t | y_<t), shape [B,T-1], không reduce."""
    shifted_logits, shifted_labels = logits[:, :-1], labels[:, 1:]
    safe_labels = shifted_labels.masked_fill(shifted_labels.lt(0), 0)
    return torch.log_softmax(shifted_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

def masked_sum(values, mask):
    return (values * mask.to(values.dtype)).sum(dim=-1)

def response_mask(labels, prompt_lengths, pad_token_id=None):
    """Mask response positions in the shifted log-probability coordinate system."""
    b, t = labels.shape
    pos = torch.arange(t, device=labels.device).expand(b, -1)
    mask = pos >= torch.as_tensor(prompt_lengths, device=labels.device).unsqueeze(1)
    if pad_token_id is not None:
        mask &= labels.ne(pad_token_id)
    mask &= labels.ge(0)
    return mask[:, 1:]
