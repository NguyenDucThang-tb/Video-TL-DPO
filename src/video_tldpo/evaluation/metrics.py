def question_metrics(rows, reference_rows=None):
    n = len(rows); acc = sum(bool(r["correct"]) for r in rows) / n if n else 0.0
    out = {"n": n, "question_accuracy": acc, "yes_rate": sum(r.get("prediction", "").lower() == "yes" for r in rows) / n if n else 0.0}
    if reference_rows is not None:
        ref = {r["sample_id"]: bool(r["correct"]) for r in reference_rows}
        overlap = [r for r in rows if r["sample_id"] in ref]
        out["wrong_to_correct"] = sum(r["correct"] and not ref[r["sample_id"]] for r in overlap)
        out["correct_to_wrong"] = sum((not r["correct"]) and ref[r["sample_id"]] for r in overlap)
        out["delta_accuracy_pp"] = 100 * (sum(r["correct"] for r in overlap) - sum(ref[r["sample_id"]] for r in overlap)) / len(overlap) if overlap else 0.0
    return out
