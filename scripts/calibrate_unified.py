"""CPU-only calibration and combined result table from completed saved margins."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from video_tldpo.evaluation.calibration import calibrate_backend, write_summary
from video_tldpo.unified import PROTOCOL_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/g/data/hn98") / os.environ.get("USER", "dd9648") / "Video-TL-DPO")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--backends", nargs="+", choices=("qwen25", "qwen3", "internvl35"), default=["qwen3", "internvl35"])
    args = parser.parse_args()
    root = args.run_root or Path(os.environ.get("VIDEO_TLDPO_RUN_ROOT", str(args.project / "outputs" / PROTOCOL_ID)))
    try:
        reports = [calibrate_backend(args.project, root, backend) for backend in args.backends]
        write_summary(root / "results.csv", reports)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, "CALIBRATION FAILED: " + str(exc) + "\n")
    for report in reports:
        print(json.dumps({"backend": report["backend"], "thresholds": report["selection"],
                          "results": report["summary"]}, indent=2))
    print("CALIBRATION_ALL: PASS", root / "results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
