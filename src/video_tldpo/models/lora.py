def apply_lora(model, r=16, alpha=32, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    return get_peft_model(model, cfg)
