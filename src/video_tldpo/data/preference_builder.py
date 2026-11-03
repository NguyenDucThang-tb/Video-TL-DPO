"""Xây binary preference an toàn từ TRAIN only; không dùng pair basic/hallucination làm DPO pair."""
import json
from pathlib import Path

def binary_record(item):
    gt = item["gt_answer"].strip().lower()
    if gt not in {"yes", "no"}: raise ValueError("Binary mode chỉ nhận gt yes/no")
    return {"sample_id": item["sample_id"], "source_dataset": item.get("source_dataset", "VideoHallucer"),
            "subset": item.get("subset"), "split": "train", "pair_id": item.get("pair_id"),
            "video_path": item["video_path"], "question": item["question"], "gt_answer": gt,
            "chosen": gt.capitalize()+".", "rejected": ("No" if gt == "yes" else "Yes")+".",
            "chosen_target_spans": [], "rejected_target_spans": [], "evidence_frame_indices": [],
            "evidence_time_intervals": [], "evidence_bboxes": [], "annotation_source": "train_ground_truth",
            "annotation_verified": False, "condition_eligible": False,
            "frame_sampling_metadata": {"num_frames": 8, "seed": 0}}

def write_binary(items, output):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("w") as f:
        for x in items: f.write(json.dumps(binary_record(x), ensure_ascii=False) + "\n")
