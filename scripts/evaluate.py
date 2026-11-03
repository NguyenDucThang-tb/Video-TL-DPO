import argparse, json
from video_tldpo.data.inspect_dataset import inspect
p=argparse.ArgumentParser(); p.add_argument("--data-root", required=True); p.add_argument("--output", required=True); a=p.parse_args()
json.dump({"status":"IMPLEMENTED_DATA_DISCOVERY_ONLY", "dataset":inspect(a.data_root)}, open(a.output,"w"), ensure_ascii=False, indent=2)
