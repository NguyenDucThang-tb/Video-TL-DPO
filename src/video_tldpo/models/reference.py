def freeze_reference(model):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model
