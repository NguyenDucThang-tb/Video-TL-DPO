"""Validator chặn dữ liệu không đủ điều kiện vào main experiment."""
from pathlib import Path

def _span_ok(text, span):
    return (isinstance(span, dict) and isinstance(span.get("start"), int) and isinstance(span.get("end"), int)
            and 0 <= span["start"] < span["end"] <= len(text) and text[span["start"]:span["end"]] == span.get("text"))

def validate_record(r, require_video=True):
    errors = []
    for k in ("sample_id", "video_path", "question", "gt_answer", "chosen", "rejected"):
        if k not in r: errors.append(f"missing:{k}")
    if require_video and not Path(r.get("video_path", "")).is_file(): errors.append("video_not_found")
    for key, text_key in (("chosen_target_spans", "chosen"), ("rejected_target_spans", "rejected")):
        for span in r.get(key, []):
            if not _span_ok(r.get(text_key, ""), span): errors.append(f"invalid_span:{key}")
    for i in r.get("evidence_frame_indices", []):
        if not isinstance(i, int) or i < 0: errors.append("invalid_frame_index")
    if r.get("condition_eligible") and not r.get("annotation_verified"): errors.append("condition_requires_verified_annotation")
    return errors

def validate_jsonl(path, rejected_path=None, require_video=True):
    import json
    good, bad = [], []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip(): continue
        r = json.loads(line); errors = validate_record(r, require_video)
        (good if not errors else bad).append({"line": line_no, "record": r, "errors": errors})
    if rejected_path:
        Path(rejected_path).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in bad) + ("\n" if bad else ""))
    return good, bad
