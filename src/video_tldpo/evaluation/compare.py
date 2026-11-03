import json
from pathlib import Path
from .metrics import question_metrics

def aggregate(paths, output=None):
    reports = []
    for p in paths: reports.append(json.loads(Path(p).read_text()))
    if output: Path(output).write_text(json.dumps(reports, ensure_ascii=False, indent=2))
    return reports
