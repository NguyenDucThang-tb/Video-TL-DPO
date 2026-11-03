import json
from pathlib import Path
from .videohallucer import discover

def inspect(root, output=None):
    rows = discover(root)
    summary = {"root": str(root), "records": len(rows), "videos": len({r["video_path"] for r in rows}),
               "subsets": sorted({r["subset"] for r in rows}), "answers": {},
               "annotation_files": len(list(Path(root).glob("*/*.json")))}
    for r in rows: summary["answers"][r["gt_answer"]] = summary["answers"].get(r["gt_answer"], 0) + 1
    if output: Path(output).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
