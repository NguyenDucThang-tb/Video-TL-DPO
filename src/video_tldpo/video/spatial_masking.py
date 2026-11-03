import numpy as np

def mask_boxes(frames, boxes_by_frame, mode="zero"):
    out = np.asarray(frames).copy()
    for fi, boxes in boxes_by_frame.items():
        if fi < 0 or fi >= len(out): raise IndexError(fi)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            h, w = out[fi].shape[:2]
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h): raise ValueError(f"bbox không hợp lệ: {box}")
            out[fi, y1:y2, x1:x2] = 0 if mode == "zero" else out[fi, y1:y2, x1:x2].mean(axis=(0,1), keepdims=True)
    return out
