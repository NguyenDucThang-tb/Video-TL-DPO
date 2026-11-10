"""Read authoritative split membership; never infer membership from an eval CSV."""
import json
from pathlib import Path
import re

from video_tldpo.evaluation.video_validation import file_sha256, read_jsonl


def sample_id(row):
    value = row.get("sample_id", row.get("id", row.get("key")))
    if value is None:
        raise ValueError("Missing sample_id/id/key; available fields: " + repr(sorted(row)))
    value = str(value)
    if value.startswith("eh/"):
        match = re.search(r"(entire|mix|interleave|misleading)[_/]?(\d+)", value)
        qi = re.search(r"\|(\d+)$", value) or re.search(r"/(\d+)(?:\|eh)?$", value)
        if not match or not qi:
            raise ValueError("Unsupported EventHallusion sample ID: " + value)
        subset = "mix" if match.group(1) == "interleave" else match.group(1)
        return "eh/{}/{}_{:03d}|{}".format(subset, subset, int(match.group(2)), int(qi.group(1)))
    return value


def normalize(row):
    sid = sample_id(row)
    pair = row.get("pair_id", sid.rsplit("|", 1)[0])
    subset = row.get("subset", str(pair).split("/")[0])
    video = row.get("video_path", row.get("video"))
    answer = str(row.get("gt_answer", row.get("answer", ""))).strip().rstrip(".").lower()
    if answer not in ("yes", "no") or not video or not row.get("question"):
        raise ValueError("Invalid video/question/binary answer for " + sid)
    if not sid.startswith("eh/"):
        if sid != str(pair) + "|" + sid.rsplit("|", 1)[-1]:
            raise ValueError("Sample/pair ID mismatch: " + sid)
        if sid.rsplit("|", 1)[-1] not in ("basic", "hallucination"):
            raise ValueError("Missing basic/hallucination role: " + sid)
    result = dict(row, sample_id=sid, pair_id=str(pair), subset=subset,
                  video_path=str(video), question=row["question"], gt_answer=answer)
    if "split" in row:
        label = str(row["split"]).lower()
        result["split"] = label[3:] if label.startswith("vh_") else label
    return result


def split_ids(path):
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("split.json must be an object containing pair-ID lists")
    raw = raw.get("pair_ids", raw.get("splits", raw))
    if not isinstance(raw, dict):
        raise ValueError("Canonical split container must be an object")
    result = {}
    for name, count in (("train", 150), ("val", 50), ("test", 155)):
        ids = raw.get(name, raw.get(name + "_ids", raw.get(name + "_pair_ids")))
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise ValueError("Unsupported split.json schema: need train/val/test pair-ID lists")
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError("Wrong canonical split count/duplicates: " + name)
        result[name] = ids
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(result[a]) & set(result[b]):
            raise ValueError("Canonical pair leakage: " + a + "/" + b)
    return result


def video_key(row):
    parts = Path(row["video_path"]).parts
    subset = row["subset"]
    return "/".join(parts[parts.index(subset):]) if subset in parts else Path(row["video_path"]).name


def _train_fields(row):
    question = row.get("question", row.get("instruction", row.get("query")))
    answer = row.get("gt_answer", row.get("answer", row.get("label",
                     row.get("output", row.get("response")))))
    video = row.get("video_path", row.get("video", row.get("video_file")))
    if not isinstance(question, str) or not isinstance(answer, str) or not video:
        raise ValueError("Unsupported canonical sft.jsonl row schema; expected question/instruction, answer/label, video/video_path; fields=" + repr(sorted(row)))
    answer = answer.strip().rstrip(".").lower()
    if answer not in ("yes", "no"):
        raise ValueError("Non-binary answer in canonical sft.jsonl: " + repr(answer))
    return question, answer, str(video)


def _canonical_train(path, train_ids, local_rows, check_files):
    """Use OPA's sft.jsonl as the actual train source and bind rows to fixed pair IDs."""
    raw_rows = read_jsonl(path)
    expected = {pid + "|" + role for pid in train_ids for role in ("basic", "hallucination")}
    by_content = {}
    by_id = {row["sample_id"]: row for row in local_rows}
    for local in local_rows:
        key = (local["question"].strip(), local["gt_answer"], video_key(local))
        by_content.setdefault(key, []).append(local)
    selected = []
    for raw in raw_rows:
        question, answer, video = _train_fields(raw)
        candidate = None
        try:
            candidate = normalize(raw)
        except ValueError:
            # Some SFT exports contain QA fields but omit sample/pair IDs. Bind
            # those rows only through a unique question+label+video match.
            pass
        if candidate is not None and candidate["sample_id"] in expected:
            match = candidate
        else:
            parts = Path(video).parts
            subset = next((s for s in ("temporal", "object_relation") if s in parts), None)
            key = (question.strip(), answer,
                   "/".join(parts[parts.index(subset):]) if subset else Path(video).name)
            matches = by_content.get(key, [])
            if len(matches) != 1:
                raise ValueError("Cannot uniquely bind canonical sft.jsonl row to authoritative train pair: " + repr(key))
            match = dict(matches[0], video_path=video)
        local = by_id.get(match["sample_id"])
        if local is None or match["sample_id"] not in expected:
            raise ValueError("Canonical sft.jsonl row is outside authoritative train split")
        if (question.strip() != local["question"].strip() or answer != local["gt_answer"] or
                video_key(dict(local, video_path=video)) != video_key(local)):
            raise ValueError("Canonical sft.jsonl question/answer/video differs for " + match["sample_id"])
        match = dict(local, video_path=video, split="train")
        if check_files and not Path(video).is_file():
            raise ValueError("Missing canonical train video: " + video)
        selected.append(match)
    if len(selected) != 300 or {r["sample_id"] for r in selected} != expected:
        raise ValueError("Canonical sft.jsonl must contain exactly the 300 authoritative train QA records")
    return selected


def load_vh(artifacts, canonical_split, canonical_eval, check_files=True, canonical_train=None):
    ids = split_ids(canonical_split)
    result = {}
    for split, file in (("train", "train_binary.jsonl"), ("val", "val_binary.jsonl"),
                        ("test", "test_records.jsonl")):
        rows = [normalize(r) for r in read_jsonl(Path(artifacts) / file)]
        wanted = {pid + "|" + role for pid in ids[split] for role in ("basic", "hallucination")}
        if len(rows) != len(wanted) or {r["sample_id"] for r in rows} != wanted:
            raise ValueError("Local records differ from canonical split membership: " + split)
        if any(r.get("split") != split for r in rows):
            raise ValueError("Incorrect local split labels: " + split)
        if {r["subset"] for r in rows} != {"temporal", "object_relation"}:
            raise ValueError("Both VideoHallucer subsets are required in " + split)
        for r in rows:
            expected = "yes" if r["sample_id"].endswith("|basic") else "no"
            if r["gt_answer"] != expected:
                raise ValueError("Incorrect basic/hallucination answer: " + r["sample_id"])
            if check_files and not Path(r["video_path"]).is_file():
                raise ValueError("Missing video: " + r["video_path"])
        for pid in ids[split]:
            if len({str(Path(r["video_path"]).resolve()) for r in rows if r["pair_id"] == pid}) != 1:
                raise ValueError("Basic/hallucination video mismatch: " + pid)
        result[split] = rows
    if canonical_train is not None:
        result["train"] = _canonical_train(canonical_train, ids["train"], result["train"], check_files)
    # Validate each actual val/test membership, question, answer and video reference.
    canonical = [normalize(r) for r in read_jsonl(canonical_eval)]
    index = {r["sample_id"]: r for r in canonical}
    expected_ids = {r["sample_id"] for split in ("val", "test") for r in result[split]}
    if len(index) != len(canonical) or set(index) != expected_ids:
        raise ValueError("eval_vh.jsonl must contain exactly the canonical val + test samples")
    for split in ("val", "test"):
        for r in result[split]:
            c = index[r["sample_id"]]
            if c.get("split", split) != split or any(r[k] != c[k] for k in ("question", "gt_answer")) \
                    or video_key(r) != video_key(c):
                raise ValueError("Canonical sample metadata differs: " + r["sample_id"])
    videos = {s: {str(Path(r["video_path"]).resolve()) for r in rows} for s, rows in result.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if videos[a] & videos[b]:
            raise ValueError("Video leakage: " + a + "/" + b)
    fingerprints = {"canonical_split": file_sha256(canonical_split),
                    "canonical_eval": file_sha256(canonical_eval)}
    if canonical_train is not None:
        fingerprints["canonical_train"] = file_sha256(canonical_train)
    for split, rows in result.items():
        by_id = {r["sample_id"]: r for r in rows}
        result[split] = [by_id[pid + "|" + role] for pid in ids[split] for role in ("basic", "hallucination")]
        fingerprints[split] = file_sha256(Path(artifacts) / {
            "train": "train_binary.jsonl", "val": "val_binary.jsonl", "test": "test_records.jsonl"}[split])
    return result, fingerprints
