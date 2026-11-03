"""Reader tương thích cấu trúc VideoHallucer của Adaptive-SEED."""
import json
from pathlib import Path

def discover(root):
    root = Path(root)
    if not root.is_dir(): raise FileNotFoundError(f"Dataset root không tồn tại: {root}")
    rows = []
    for subset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for ann in sorted(subset_dir.glob("*.json")):
            data = json.loads(ann.read_text())
            if not isinstance(data, list): continue
            for ex_i, ex in enumerate(data):
                for role in ("basic", "hallucination"):
                    q = ex.get(role, {})
                    video = q.get("video")
                    if video and not Path(video).is_absolute(): video = str(root / subset_dir.name / Path(video).name)
                    rows.append({"source_dataset":"VideoHallucer", "subset":subset_dir.name, "role":role,
                                 "pair_id":f"{subset_dir.name}/{ann.stem}/{ex_i}", "video_path":video,
                                 "question":q.get("question",""), "gt_answer":str(q.get("answer","")).strip().lower()})
    return rows
