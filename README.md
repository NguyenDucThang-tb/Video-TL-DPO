# Video-TL-DPO

Research on video hallucination preference optimization with Qwen2.5-VL-7B,
Qwen3-VL-8B and InternVL3.5-8B.

New baseline runs use the shared [unified experiment protocol](docs/UNIFIED_PROTOCOL.md),
matching the supplied [configuration reference](docs/UNIFIED_CONFIG_REFERENCE.md).
The executable settings live in `src/video_tldpo/unified.py`.

```bash
cd /g/data/hn98/$USER/Video-TL-DPO
bash scripts/submit_unified_pipeline.sh --backends qwen3 internvl35 --preflight-only
bash scripts/submit_unified_pipeline.sh --backends qwen3 internvl35 --walltime 00:30:00
```

Use `qwen25` or `internvl35` for the other backbones. PBS extracts the existing
Conda archive into JobFS and writes persistent artifacts to GData. Preflight
validates canonical sample membership. Base checks require the reference aggregate
counts before reporting, with per-sample reference margins as diagnostics.
All submission commands are manual; acceptance is not completion.

The [v3 runbook](docs/UNIFIED_V3_RUNBOOK.md) explains paths and job dependencies.
Protocol v3 evaluates VH_VAL for both base and adapter, then fits each threshold
using only VH_VAL. It applies that fixed threshold to VH_TEST and EventHallusion
and exports both raw and calibrated tables to `outputs/unified_adaptive_seed_v3/results.csv`.
Earlier v2 outputs are preserved. Each fresh run starts from the original model
and a new LoRA, not another method's trained adapter.

The pipeline above trains Vanilla DPO with the shared student budget. The
Target Generation and Target Condition losses remain research components;
`scripts/train.py` and TL-specific PBS templates are unfinished scaffolding.
Previous generation-based evaluation scores and r8/lr1e-5 checkpoints belong to
the old protocol and cannot be resumed into the unified run.

Local contract checks, without optional ML packages:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_unified*.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_calibration.py' -v
```

Model execution, numerical base reproduction and GPU resume require real Gadi
logs. Existing historical Gadi success does not verify this changed pipeline.
