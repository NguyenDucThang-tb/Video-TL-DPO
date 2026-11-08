# Qwen3-VL and InternVL3.5 Vanilla DPO

This is a separate **Vanilla DPO LoRA baseline** using the existing fixed
VideoHallucer split. It does not implement or claim the paper's Target
Generation Loss or Target Condition Loss. It keeps the 300 train records, 100
validation records, 310 test records, and the same 409-question EventHallusion
evaluation used by the Qwen2.5-VL experiment.

## Model arms

| Backend | Checkpoint | Video representation |
| --- | --- | --- |
| `qwen3` | `Qwen/Qwen3-VL-8B-Instruct` | Eight uniformly sampled frames passed through Qwen3's video processor, with original FPS, duration, and source frame indices retained for temporal metadata. |
| `internvl35` | `OpenGVLab/InternVL3_5-8B-Instruct` | Up to eight ordered RGB frames, resized to 448×448 and encoded as one image tile per frame. |

For both arms, the user prompt ends with `Answer yes or no.` Chosen and rejected
sequences use the same decoded frames and prompt. The DPO objective, beta,
response-only causal log-probability sum, and train/validation/test assignments
match the existing Vanilla DPO run. LoRA is selected from exact linear layer
names inside `language_model`; it must not target the visual encoder or
InternVL's `mlp1` projector.

The InternVL video-as-ordered-frames route follows the repository's existing
InternVL wrapper. It is not claimed to be an official InternVL temporal-video
training recipe. Both model-specific frame protocols are recorded in run
metadata and must be reported alongside scores.

## Gadi asset setup

The setup PBS job creates one Conda environment, packs it into the group
environment archive, downloads both model snapshots into the user's GData model
directory, and records their immutable Hugging Face revisions. It intentionally
leaves the temporary build environment in place until the JobFS smoke succeeds.
Its Conda package cache is redirected away from the shared read-only Conda base
and removed after the environment is packed.

```bash
cd /g/data/hn98/$USER/Video-TL-DPO-code-download
git pull --ff-only origin main

PROJECT=/g/data/hn98/$USER/Video-TL-DPO
cp scripts/train_videohallucer_backend.py \
   scripts/smoke_videohallucer_backend.py \
   scripts/preflight_vlm8b.py \
   scripts/submit_vlm8b_setup.sh \
   scripts/submit_vlm8b_chain.sh \
   scripts/submit_vlm8b_eval.sh \
   "$PROJECT/scripts/"
cp src/video_tldpo/models/vlm_backends.py "$PROJECT/src/video_tldpo/models/"
cp src/video_tldpo/evaluation/video_validation.py \
   src/video_tldpo/evaluation/eventhallusion.py \
   "$PROJECT/src/video_tldpo/evaluation/"
cp scripts/evaluate_videohallucer_validation.py "$PROJECT/scripts/"
cp pbs/setup_vlm8b_assets.pbs \
   pbs/train_videohallucer_backend.pbs \
   pbs/evaluate_videohallucer_backend.pbs \
   "$PROJECT/pbs/"

cd "$PROJECT"
bash scripts/submit_vlm8b_setup.sh
```

Inspect the setup log under `/g/data/hn98/$USER/logs/`. Continue only when it
contains `SETUP_ARCHIVE_AND_MODELS: PASS`; confirm the archive and both model
folders exist. The expected archive is
`/g/data/hn98/envs/${USER}_video_vlm8b.tar.gz`. The setup writes the resolved
model commit IDs to `video_vlm8b_models.json` and to each model directory's
`model_revision.json`.

## Interactive H200 smoke test

Request one interactive Hopper GPU allocation (12 CPUs are required):

```bash
qsub -I -P hn98 -q gpuhopper \
  -l ngpus=1,ncpus=12,mem=64GB,walltime=01:00:00,jobfs=24GB \
  -l storage=gdata/hn98+scratch/jp09
```

In the allocated shell:

```bash
set -euo pipefail
PROJECT=/g/data/hn98/$USER/Video-TL-DPO
USER_ROOT=/g/data/hn98/$USER
ARCHIVE=/g/data/hn98/envs/${USER}_video_vlm8b.tar.gz
ENV_ROOT="$PBS_JOBFS/video_vlm8b_env"
mkdir -p "$ENV_ROOT"
tar -xzf "$ARCHIVE" -C "$ENV_ROOT"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
"$ENV_ROOT/bin/python" "$ENV_ROOT/bin/conda-unpack"
export PATH="$ENV_ROOT/bin:$PATH"
export CONDA_PREFIX="$ENV_ROOT" CONDA_DEFAULT_ENV="$ENV_ROOT"
export HF_HOME="$USER_ROOT/hf_cache" XDG_CACHE_HOME="$USER_ROOT/cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT/src:$PROJECT/scripts${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi

for BACKEND in qwen3 internvl35; do
  if [ "$BACKEND" = qwen3 ]; then
    MODEL="$USER_ROOT/models/Qwen3-VL-8B-Instruct"
  else
    MODEL="$USER_ROOT/models/InternVL3_5-8B-Instruct"
  fi
  "$ENV_ROOT/bin/python" "$PROJECT/scripts/smoke_videohallucer_backend.py" \
    --backend "$BACKEND" --model "$MODEL" \
    --data "$PROJECT/artifacts/train_binary.jsonl" \
    --output "$PROJECT/outputs/smoke_${BACKEND}"
done
```

Each backend must end with its own `NATIVE_VIDEO_DPO_SMOKE_*: PASS` marker. This
checks model load, native video inputs, chosen/rejected token alignment,
teacher-forced policy/reference scores, frozen reference, nonzero language-LoRA
gradients, optimizer update, adapter save, and generation. This smoke does not
verify convergence or full training. After a successful JobFS smoke, remove only
the temporary build environment `/g/data/hn98/$USER/conda_build/video_vlm8b`;
keep the packed archive and model snapshots.

## Preflight and training

Run from the operational GData project. The preflight uses stdlib Python and
checks model type/revision, weight shards, all 300/100/310 records, manifest
membership, video existence, and zero video overlap.

```bash
cd /g/data/hn98/$USER/Video-TL-DPO
python3 scripts/preflight_vlm8b.py qwen3 \
  /g/data/hn98/$USER/models/Qwen3-VL-8B-Instruct artifacts
python3 scripts/preflight_vlm8b.py internvl35 \
  /g/data/hn98/$USER/models/InternVL3_5-8B-Instruct artifacts
```

Submit one sequential, resumable one-hour chain per arm after that arm's smoke
passes:

```bash
bash scripts/submit_vlm8b_chain.sh qwen3 16
bash scripts/submit_vlm8b_chain.sh internvl35 16
```

Each chain has one active job at a time and uses `afterok`; all jobs share the
same durable checkpoint directory for that backend. Do not use an array or
submit both backends to the same output path. A submitted job is not a completed
training run: require `VANILLA_DPO_QWEN3_TRAINING: PASS` or
`VANILLA_DPO_INTERNVL35_TRAINING: PASS`, exit status zero, and a summary with
`status=COMPLETED`, `records=300`.

## Evaluation

After that backend's training summary is complete, evaluate the held-out
VideoHallucer test set and full EventHallusion set. Each script submits a base
job and then an adapter job dependent on the base job.

```bash
bash scripts/submit_vlm8b_eval.sh qwen3 videohallucer test
bash scripts/submit_vlm8b_eval.sh qwen3 eventhallusion test
bash scripts/submit_vlm8b_eval.sh internvl35 videohallucer test
bash scripts/submit_vlm8b_eval.sh internvl35 eventhallusion test
```

Expected outputs are under
`outputs/{videohallucer|eventhallusion}_{qwen3|internvl35}_vanilla_dpo_v1/`.
Use `comparison.json` only after both `*_BASE: PASS` and `*_ADAPTER: PASS` appear
and both jobs have exit status zero. Report VideoHallucer question accuracy,
VideoHallucer pair accuracy, and EventHallusion official question accuracy;
include each backend's base score, DPO score, and percentage-point delta.

## Evidence boundary

Local unit tests exercise split-independent utilities and structural checks.
They do not load either 8B checkpoint. The actual model forward, CUDA LoRA
backward, optimizer step, adapter save, full 300-record training, and evaluation
remain `NOT RUN` until the corresponding Gadi logs and artifacts are inspected.
