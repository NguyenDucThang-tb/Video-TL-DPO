# Video-TL-DPO

Research on video hallucination preference optimization with Qwen2.5-VL-7B,
Qwen3-VL-8B and InternVL3.5-8B.

New baseline runs use the shared [unified experiment protocol](docs/UNIFIED_PROTOCOL.md),
matching the supplied [configuration reference](docs/UNIFIED_CONFIG_REFERENCE.md).
The executable settings live in `src/video_tldpo/unified.py`.

```bash
cd /g/data/hn98/$USER/Video-TL-DPO
bash scripts/submit_unified.sh base qwen3 --preflight-only
bash scripts/submit_unified.sh base qwen3
# Training can run before base evaluation; run base checks before reporting comparisons.
bash scripts/submit_unified.sh train qwen3 --jobs 2
# After the unified 300-record / 38-step training completes:
bash scripts/submit_unified.sh eval qwen3
```

Use `qwen25` or `internvl35` for the other backbones. PBS extracts the existing
Conda archive into JobFS and writes persistent artifacts to GData. Base checks
validate canonical sample membership and per-sample predictions before training.
All submission commands are manual; acceptance is not completion.

The pipeline above trains Vanilla DPO with the shared student budget. The
Target Generation and Target Condition losses remain research components;
`scripts/train.py` and TL-specific PBS templates are unfinished scaffolding.
Previous generation-based evaluation scores and r8/lr1e-5 checkpoints belong to
the old protocol and cannot be resumed into the unified run.

Local contract checks, without optional ML packages:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_unified*.py' -v
```

Model execution, numerical base reproduction and GPU resume require real Gadi
logs. Existing historical Gadi success does not verify this changed pipeline.
