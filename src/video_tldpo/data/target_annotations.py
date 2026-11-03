"""Helpers cho character-span annotation và review queue; không tự nhận diện evidence."""
import re

def make_span(text, phrase, occurrence=0):
    hits = list(re.finditer(re.escape(phrase), text))
    if occurrence >= len(hits): raise ValueError(f"Không tìm thấy phrase chính xác: {phrase!r}")
    m = hits[occurrence]
    return {"start": m.start(), "end": m.end(), "text": m.group(0)}

def verified_condition_eligible(record):
    return bool(record.get("annotation_verified") and record.get("evidence_frame_indices") and
                (record.get("chosen_target_spans") or record.get("rejected_target_spans")))
