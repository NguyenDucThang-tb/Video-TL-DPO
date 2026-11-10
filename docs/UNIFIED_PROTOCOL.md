# Unified experiment protocol

The executable source of settings is `src/video_tldpo/unified.py`. All operational
PBS/submit entry points use `scripts/run_unified.py`; YAML files are snapshots,
not overrides of this contract. `UNIFIED_CONFIG_REFERENCE.md` contains the
latest user-supplied configuration snapshot. Its old venv commands are not used:
environments remain archives and are extracted only into `$PBS_JOBFS`.

Implemented: Vanilla DPO baseline with the unified student budget. This change
does not claim the target-generation/target-condition method has been trained.

## Data and inference

Preflight reads the authoritative OPA `outputs/data/split.json` and
`outputs/data/eval_vh.jsonl`. Base evaluation only needs those split/eval files.
Training and adapter evaluation additionally require
`outputs/work_binary_all/sft.jsonl`; training uses those canonical SFT rows as
the actual source, binding every row to fixed train pair IDs and checking
question, answer and video against the prepared training artifact. It validates
actual train/val/test pair membership,
each val/test question, label and video reference, requires 300/100/310 records
and both subsets, and checks video separation. No new split is generated.
Accepted split schemas are `pair_ids`,
`splits`, or root objects with `train`, `val`, `test` (also `_ids`/`_pair_ids`)
string lists. Eval rows accept `sample_id`/`id`/`key`, `video_path`/`video`,
`gt_answer`/`answer`. Unknown schemas fail explicitly; inspect the actual files
rather than assuming equivalence from counts or an all-splits prediction CSV.

Dataset revision in the contract is the required HF revision; file SHA256s record
actual canonical files and artifacts. The revision claim itself cannot be proven
from an unversioned video directory. Source data and weights are never downloaded
by this workflow.

Each video is decoded with decord; exactly 8 linspace/round indices are used,
including repeated indices for short clips. Frames become RGB PIL images and
are resized to 448x448 bicubic. Sampled FPS is exactly 8 / (N / avg_fps).
Qwen2.5 receives `fps=[sampled_fps]`. Qwen3 receives 8-frame metadata and indices
0..7. InternVL uses one 256-token image per frame, the original system template,
ImageNet normalization and image_flags=1.

The prompt ends with `\nAnswer with yes or no.`. Evaluation is a single forward,
not generation: margin > 0 predicts yes, ties predict no. All three backbones
score the `Yes` and `No` token IDs obtained from the first token of
`tokenizer.encode(word, add_special_tokens=False)`. This includes Qwen3 even
though its generated text is usually lowercase. Qwen attention is explicitly SDPA;
InternVL disables FlashAttention and uses eager language attention. Qwen3 and
InternVL require Transformers 4.57.6.

## Training and checkpoints

Use r16/alpha32/dropout0.05/bias-none, the exact attention+projector regexes in
the reference, AdamW lr=1e-4/weight_decay=0, clip=1, cosine and seed 0. Model
weights are frozen outside LoRA; the reference is separately loaded and frozen.
Qwen2.5 must have 10,395,648 trainable parameters; InternVL must have 15,597,568.
The DPO scores are only the two answer-token log probabilities at the end of
the prompt; punctuation and EOS are not training targets. Beta is 0.1.

One seeded shuffle of 300 train records, microbatch 1, accumulation 8 produces
38 optimizer steps. Warmup is floor(0.05 * 300 / 8) = 1 step, matching
Adaptive-SEED. The last group has 4 records and, as in Adaptive-SEED, each loss
is divided by 8. Checkpoints are saved only after complete accumulation groups.
They include optimizer, scheduler, sample order/cursor, step counter, and RNG.
Contracts also fingerprint the runtime source, processor files, model identity
and canonical data, so changing code or data invalidates old resume state.
A one-hour job pauses cleanly; an afterok chain continues the same run.

Canonical base counts are Qwen2.5 `230/310`, `78/155`, `270/409`; Qwen3
`235/310`, `89/155`, `264/409`; InternVL `212/310`, `66/155`, `198/409`.
Outputs live under `outputs/unified_adaptive_seed_v2/<backend>/`. Old runs are
preserved, and changed contracts/legacy checkpoints are refused. The old
resumable module retains legacy helper functions for artifact regressions;
its command-line entry now routes to the unified runner. The generic
`scripts/train.py` and TL-specific PBS templates are unfinished research
scaffolding and must not be used as evidence of unified training.

## Gadi sequence (manual)

After pushing/syncing the code, work in `/g/data/hn98/$USER/Video-TL-DPO`.
Mount gdata/hn98, scratch/hn98 and scratch/jp09. Defaults use the existing model
directories and env archives. No permanent venv or unpacked Conda env is created.
Canonical EventHallusion defaults to `/scratch/hn98/$USER/runs/exp1/EventHallusion`
with its questions files, all 409 questions. If using a relocated copy, set
`VIDEO_TLDPO_EVENT_ROOT` and optionally `VIDEO_TLDPO_EVENT_QUESTIONS` before
preflight. These variables are exported into PBS jobs.

First validate and submit ONLY untrained models (six independent GPU jobs):

```bash
cd /g/data/hn98/$USER/Video-TL-DPO
for BACKEND in qwen25 qwen3 internvl35; do
  bash scripts/submit_unified.sh base "$BACKEND" --preflight-only || break
  bash scripts/submit_unified.sh base "$BACKEND" || break
done
```

Each base must match the specified question and pair counts. If the corresponding
canonical margin file exists, per-sample margin differences are recorded as
diagnostics; they do not replace the required aggregate-count check. If counts
differ, use those margins to locate preprocessing or token-scoring differences.
Missing diagnostics do not block base scoring. No offsets or thresholds are
adjusted to fit the target. A separate base check is needed for VH_TEST and
EH_ALL.

Training may be submitted after its data/model preflight passes; base evaluation
does not gate optimizer training. Run the base jobs before reporting comparisons
so the published baseline and adapter use the validated shared protocol.

```bash
bash scripts/submit_unified.sh train qwen3 --preflight-only
bash scripts/submit_unified.sh train qwen3 --jobs 2
```

Replace qwen3 with qwen25 or internvl35. `--jobs 2` is a sequential one-hour
checkpoint-resume chain, not independent training shards. The default is one
job to avoid queuing many empty jobs. Completion requires Exit_status=0 and
`train_summary.json` with COMPLETED, records=300, optimizer_steps=38.

Then evaluate the final adapter on both benchmarks:

```bash
bash scripts/submit_unified.sh eval qwen3 --preflight-only
bash scripts/submit_unified.sh eval qwen3
```

Base files have `base/margins.jsonl`, `metrics.json`, `base_check.json`; adapter
files have `adapter/margins.jsonl`, `metrics.json`; completed comparisons are
`<dataset>_test/comparison.json`. Evaluation can resume saved margins; a partial
evaluation returns code 3 and requires resubmitting that command after it ends.
Use `--dataset videohallucer` or `--dataset eventhallusion` to rerun only one.
Both complete metrics and PBS exit statuses must be checked before reporting.
Adapter inference refuses the old training budgets; old checkpoints cannot be
resumed into a new unified run.

Canonical locations can be overridden with VIDEO_TLDPO_CANONICAL_SPLIT,
VIDEO_TLDPO_CANONICAL_EVAL and VIDEO_TLDPO_CANONICAL_TRAIN. Model/env overrides
remain VIDEO_TLDPO_MODEL and VIDEO_TLDPO_ARCHIVE. Use `--backend` + the CLI
`--reference-margins` option to inspect a different canonical reference explicitly.

## Local evidence boundary

Contract regressions: `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_unified*.py' -v`.
The numeric CPU resume regression is skipped unless Torch, NumPy and safetensors
are installed; run it again from the packed Gadi environment.
The tests check canonical split membership/content, repeated short-video
sampling, LoRA selection, budget rounding and prediction-level base validation. They do
not load models. GPU forward/backward, numerical base reproduction, parameter
counts, real canonical schemas and checkpoint resume remain to be verified by
Gadi logs. Existing historical metrics remain results of the old protocol.
