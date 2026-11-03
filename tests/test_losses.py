import torch
from video_tldpo.losses.log_probs import token_log_probs
from video_tldpo.losses.vanilla_dpo import dpo_loss
from video_tldpo.losses.target_generation import target_generation_loss

def test_token_log_probs_and_dpo_are_finite():
    logits=torch.randn(2,5,11,requires_grad=True); labels=torch.tensor([[1,2,3,4,5],[2,3,4,5,6]])
    x=token_log_probs(logits,labels); assert x.shape==(2,4)
    loss,_=dpo_loss(x.sum(1),x.sum(1)-1,x.sum(1)-.1,x.sum(1)-.9); loss.backward(); assert torch.isfinite(logits.grad).all()

def test_target_generation_ignores_unrelated_tokens():
    z=torch.zeros(1,4,requires_grad=True); m=torch.tensor([[0.,1.,0.,0.]])
    loss,_=target_generation_loss(z,z-1,z-.1,z-.9,m,m); before=loss.item()
    z.data[0,0]=100; assert loss.item()==before

def test_target_gradient_nonzero():
    z=torch.zeros(1,3,requires_grad=True); m=torch.tensor([[1.,0.,1.]])
    refc=torch.zeros(1,3); refr=torch.zeros(1,3)
    loss,_=target_generation_loss(z,torch.zeros(1,3),refc,refr,m,m); loss.backward(); assert z.grad.abs().sum()>0
