"""Submit fresh training, raw base/adapter val+test evaluation and CPU calibration.

This is a submit/preflight tool only. It never loads models or extracts an env
on the login node. Each GPU job extracts its archive into PBS_JOBFS.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from video_tldpo.unified import LORA_SCOPES, PROTOCOL_ID


def build_plan(backends, train_jobs):
    plan, final = [], []
    for backend in backends:
        previous = None
        for number in range(train_jobs):
            key = f"{backend}/train/{number + 1}"
            plan.append(dict(key=key, backend=backend, command="train", variant="base",
                dataset="videohallucer", split="test", dependencies=[previous] if previous else []))
            previous = key
        for dataset, split in (("videohallucer", "val"), ("videohallucer", "test"), ("eventhallusion", "test")):
            base = f"{backend}/base/{dataset}/{split}"
            adapter = f"{backend}/adapter/{dataset}/{split}"
            plan.append(dict(key=base, backend=backend, command="evaluate", variant="base",
                dataset=dataset, split=split, dependencies=[]))
            plan.append(dict(key=adapter, backend=backend, command="evaluate", variant="adapter",
                dataset=dataset, split=split, dependencies=[previous, base]))
            final.append(adapter)
    plan.append(dict(key="calibration", command="calibrate", dependencies=final))
    return plan


def walltime_seconds(value):
    match = re.fullmatch(r"(\d{2}):([0-5]\d):([0-5]\d)", value)
    if not match:
        raise argparse.ArgumentTypeError("Use HH:MM:SS")
    hours, minutes, seconds = map(int, match.groups())
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 300:
        raise argparse.ArgumentTypeError("Walltime must exceed five minutes")
    return total


def call(command, env, capture=False, timeout=120):
    result = subprocess.run(command, env=env, check=True, universal_newlines=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None, timeout=timeout)
    return result.stdout.strip() if capture else None


def preflight(args, env):
    if not args.resume and (args.run_root / "pipeline_jobs.json").exists():
        raise ValueError("This run was already submitted; use a new --run-name or --resume after all jobs stop")
    if env.get("VIDEO_TLDPO_OUT"):
        raise ValueError("Unset VIDEO_TLDPO_OUT: the pipeline manages a separate directory for each stage")
    if len(args.backends) > 1 and env.get("VIDEO_TLDPO_MODEL"):
        raise ValueError("Unset VIDEO_TLDPO_MODEL when submitting multiple backends")
    for backend in args.backends:
        destination = args.run_root / backend
        if not args.resume and any(destination.glob("**/run_meta.json")):
            raise ValueError("Run output already exists; choose a new --run-name or --resume")
        if not args.resume and (destination / "train/trainer_state").exists():
            raise ValueError("Training state already exists; choose a new --run-name or --resume")
        archive = env.get("VIDEO_TLDPO_ARCHIVE", str(Path("/g/data/hn98/envs") /
            (env["USER"] + ("_qwen25_vtdpo_v2.tar.gz" if backend == "qwen25" else "_video_vlm8b.tar.gz"))))
        if not Path(archive).is_file():
            raise ValueError("Missing environment archive: " + archive)
        for command, dataset, split in (("train", "videohallucer", "test"),
                ("evaluate", "videohallucer", "val"), ("evaluate", "videohallucer", "test"),
                ("evaluate", "eventhallusion", "test")):
            cli = [sys.executable, str(ROOT / "scripts/run_unified.py"), command,
                   "--backend", backend, "--project", str(args.project),
                   "--run-root", str(args.run_root), "--dataset", dataset, "--split", split, "--preflight-only"]
            for variable, flag in (("VIDEO_TLDPO_MODEL", "--model"),
                                   ("VIDEO_TLDPO_EVENT_QUESTIONS", "--questions-dir"),
                                   ("VIDEO_TLDPO_EVENT_ROOT", "--dataset-root")):
                if env.get(variable):
                    cli.extend([flag, env[variable]])
            call(cli, env)


def submit(args, env, plan):
    args.run_root.mkdir(parents=True, exist_ok=True)
    with (args.run_root / ".submission.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another submission is modifying this run")
        manifest_path = args.run_root / "pipeline_jobs.json"
        if manifest_path.exists():
            if not args.resume:
                raise ValueError("Pipeline already submitted; refusing duplicates")
            old = json.loads(manifest_path.read_text())
            for job in old.get("jobs", []):
                result = subprocess.run(["qstat", "-xf", job["job_id"]], env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
                if result.returncode != 0:
                    raise ValueError("Cannot verify old job state; check qstat before resubmitting: " + job["job_id"])
                if not re.search(r"job_state\s*=\s*F\b", result.stdout):
                    raise ValueError("An old pipeline job is still active: " + job["job_id"])
        logdir = args.project / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ids_file = logdir / f"unified_pipeline_{stamp}_{os.getpid()}.ids"
        manifest = {"protocol": PROTOCOL_ID, "run_root": str(args.run_root),
            "backends": args.backends, "walltime": args.walltime,
            "lora_scope": env.get("VIDEO_TLDPO_LORA_SCOPE", "attention_projector"),
            "job_ids_file": str(ids_file), "jobs": []}
        ids = {}
        for node in plan:
            job_env = dict(env, VIDEO_TLDPO_PROJECT=str(args.project),
                           VIDEO_TLDPO_RUN_ROOT=str(args.run_root))
            if node["command"] == "calibrate":
                job_env["VIDEO_TLDPO_BACKENDS"] = ":".join(args.backends)
                exports = ["VIDEO_TLDPO_PROJECT", "VIDEO_TLDPO_RUN_ROOT", "VIDEO_TLDPO_BACKENDS"]
                command = ["qsub", "-r", "y", "-N", "vtdpo3_calibration"]
                script = args.project / "pbs/calibrate_unified.pbs"
            else:
                backend = node["backend"]
                job_env.update(VIDEO_TLDPO_BACKEND=backend, VIDEO_TLDPO_COMMAND=node["command"],
                    VIDEO_TLDPO_EVAL_VARIANT=node["variant"], VIDEO_TLDPO_EVAL_DATASET=node["dataset"],
                    VIDEO_TLDPO_EVAL_SPLIT=node["split"], VIDEO_TLDPO_JOB_SECONDS=str(args.seconds))
                job_env["VIDEO_TLDPO_ARCHIVE"] = env.get("VIDEO_TLDPO_ARCHIVE", str(Path("/g/data/hn98/envs") /
                    (env["USER"] + ("_qwen25_vtdpo_v2.tar.gz" if backend == "qwen25" else "_video_vlm8b.tar.gz"))))
                exports = [key for key in job_env if key.startswith("VIDEO_TLDPO_") and key != "VIDEO_TLDPO_OUT"]
                mode = "train" if node["command"] == "train" else "base" if node["variant"] == "base" else "eval"
                prefix = "vtdpo3_la" if env.get("VIDEO_TLDPO_LORA_SCOPE") == "language_attention" else "vtdpo3"
                name = f"{prefix}_{backend}_{mode}_{node['dataset']}_{node['split']}"
                command = ["qsub", "-r", "y", "-l", "walltime=" + args.walltime, "-N", name]
                script = args.project / "pbs/run_unified.pbs"
            if node["dependencies"]:
                command.extend(["-W", "depend=afterok:" + ":".join(ids[key] for key in node["dependencies"])])
            command.extend(["-v", ",".join(sorted(exports)), str(script)])
            job = call(command, job_env, capture=True, timeout=60)
            if not re.fullmatch(r"\d+(?:\[.*\])?\.[A-Za-z0-9_.-]+", job):
                raise ValueError("qsub returned an unexpected response; inspect qstat before retrying: " + repr(job))
            ids[node["key"]] = job
            manifest["jobs"].append(dict(node, job_id=job))
            with ids_file.open("a") as stream:
                stream.write(job + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(manifest, indent=2) + "\n")
            temporary.replace(manifest_path)
            print(node["key"] + "=" + job, flush=True)
        print("job_ids_file:", ids_file)
        print("results_after_jobs_finish:", args.run_root / "results.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(os.environ.get("VIDEO_TLDPO_PROJECT", "/g/data/hn98/{}/Video-TL-DPO".format(os.environ.get("USER", "dd9648")))))
    parser.add_argument("--backends", nargs="+", choices=("qwen25", "qwen3", "internvl35"), default=["qwen3", "internvl35"])
    parser.add_argument("--run-name")
    parser.add_argument("--lora-scope", choices=LORA_SCOPES,
                        default=os.environ.get("VIDEO_TLDPO_LORA_SCOPE", "attention_projector"))
    parser.add_argument("--walltime", default="00:30:00")
    parser.add_argument("--train-jobs", type=int, choices=range(1, 17), default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = PROTOCOL_ID + ("_language_attention" if args.lora_scope == "language_attention" else "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name) or args.run_name in (".", ".."):
        parser.error("run-name must be a directory name")
    if len(set(args.backends)) != len(args.backends):
        parser.error("Duplicate backends")
    try:
        args.seconds = walltime_seconds(args.walltime)
        args.run_root = args.project / "outputs" / args.run_name
        env = dict(os.environ, VIDEO_TLDPO_RUN_ROOT=str(args.run_root), VIDEO_TLDPO_LORA_SCOPE=args.lora_scope)
        print("lora_scope: {} run_root: {}".format(args.lora_scope, args.run_root), flush=True)
        preflight(args, env)
        plan = build_plan(args.backends, args.train_jobs)
        print("pipeline_plan: GPU_jobs={} CPU_calibration_jobs=1".format(len(plan) - 1), flush=True)
        if args.preflight_only:
            print("preflight_only: no jobs submitted")
            return 0
        submit(args, env, plan)
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, "PIPELINE FAILED: {}. Check qstat and pipeline_jobs.json before retrying.\n".format(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
