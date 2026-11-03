import numpy as np

def mask_temporal(frames, evidence_indices, mode="blur", seed=0, noise_std=25.0):
    """Corrupt selected frames deterministically; không đổi shape/count/order."""
    out = np.asarray(frames).copy()
    rng = np.random.default_rng(seed)
    for i in sorted(set(evidence_indices)):
        if i < 0 or i >= len(out): raise IndexError(i)
        if mode == "noise": out[i] = np.clip(out[i].astype(np.float32) + rng.normal(0, noise_std, out[i].shape), 0, 255)
        elif mode == "zero": out[i] = 0
        elif mode == "blur":
            # dependency-free local blur, suitable for deterministic CPU tests
            x = out[i].astype(np.float32); p = np.pad(x, ((1,1),(1,1),(0,0)), mode="edge")
            out[i] = (p[:-2,1:-1] + p[2:,1:-1] + p[1:-1,:-2] + p[1:-1,2:] + x) / 5
        else: raise ValueError(f"mask mode không hỗ trợ: {mode}")
    return np.rint(out).astype(np.asarray(frames).dtype)
