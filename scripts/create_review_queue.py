import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); a=p.parse_args()
rows=[]
for line in Path(a.input).read_text().splitlines():
    if line.strip():
        r=json.loads(line); r["review_status"]="pending"; r["review_notes"]=""; rows.append(r)
Path(a.output).write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+("\n" if rows else ""))
print(f"created review queue: {len(rows)}")
