import argparse
from video_tldpo.evaluation.compare import aggregate
p=argparse.ArgumentParser(); p.add_argument("inputs", nargs="+"); p.add_argument("--output", required=True); a=p.parse_args(); aggregate(a.inputs,a.output)
