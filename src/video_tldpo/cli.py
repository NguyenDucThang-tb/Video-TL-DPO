import argparse, json
from pathlib import Path
from .data.inspect_dataset import inspect
from .data.videohallucer import discover
from .data.preference_builder import write_binary
from .data.validator import validate_jsonl

def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inspect-data"); a.add_argument("--root", required=True); a.add_argument("--output")
    a = sub.add_parser("build-binary"); a.add_argument("--root", required=True); a.add_argument("--output", required=True)
    a = sub.add_parser("validate"); a.add_argument("--input", required=True); a.add_argument("--rejected", required=True); a.add_argument("--allow-missing-video", action="store_true")
    args = p.parse_args()
    if args.cmd == "inspect-data": print(json.dumps(inspect(args.root, args.output), ensure_ascii=False, indent=2))
    elif args.cmd == "build-binary": write_binary([r for r in discover(args.root) if r["role"] == "basic"], args.output); print(args.output)
    else:
        good,bad = validate_jsonl(args.input, args.rejected, not args.allow_missing_video); print(json.dumps({"valid":len(good),"rejected":len(bad)}))

if __name__ == "__main__": main()
