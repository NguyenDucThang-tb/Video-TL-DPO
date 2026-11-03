import json
from pathlib import Path
from video_tldpo.data.validator import validate_record
from video_tldpo.data.splits import video_disjoint_split, assert_disjoint

def test_span_validator_requires_exact_text(tmp_path):
    p=tmp_path/'v.mp4'; p.write_bytes(b'fixture')
    r={"sample_id":"x","video_path":str(p),"question":"q","gt_answer":"yes","chosen":"A dog.","rejected":"A cat.","chosen_target_spans":[{"start":2,"end":5,"text":"dog"}],"rejected_target_spans":[],"condition_eligible":False}
    assert validate_record(r)==[]
    r["chosen_target_spans"][0]["text"]="cat"; assert "invalid_span:chosen_target_spans" in validate_record(r)

def test_split_is_video_disjoint():
    rows=[{"video_path":f"v{i}","sample_id":str(i)} for i in range(20)]
    s=video_disjoint_split(rows,seed=4); assert_disjoint(s); assert sum(map(len,s.values()))==20
