# Video-TL-DPO

Project độc lập chuyển TL-DPO từ image sang video với Qwen2.5-VL-7B-Instruct. Code không sửa Adaptive-SEED và không copy video.

## Hiện trạng môi trường

Mặc định Gadi: dataset `/scratch/jp09/dd9648/datasets_video_hallu`, project `/scratch/hn98/dd9648/Video-TL-DPO`. Checkout hiện tại không có `/scratch`, CUDA, `module` hoặc pip; vì vậy các lệnh GPU/PBS chưa chạy trong phiên này.

## Cài đặt

```bash
cd /scratch/hn98/dd9648/Video-TL-DPO
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
# trên GPU node: pip install -e '.[gpu]'
```

## CPU validation

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/inspect_data.py inspect-data --root /scratch/jp09/dd9648/datasets_video_hallu --output artifacts/dataset_summary.json
PYTHONPATH=src python scripts/build_preferences.py build-binary --root /scratch/jp09/dd9648/datasets_video_hallu --output artifacts/train_binary.jsonl
PYTHONPATH=src python scripts/validate_preferences.py validate --input artifacts/train_binary.jsonl --rejected artifacts/train_binary.rejected.jsonl
```

## Experiments

```bash
PYTHONPATH=src python scripts/train.py --config configs/vanilla_dpo.yaml
PYTHONPATH=src python scripts/train.py --config configs/tl_generation.yaml
PYTHONPATH=src python scripts/train.py --config configs/tl_condition.yaml
PYTHONPATH=src python scripts/train.py --config configs/tl_full.yaml
PYTHONPATH=src python scripts/aggregate_results.py outputs/*/report.json --output outputs/aggregate.json
```

The current training entrypoint deliberately stops after a GPU/model-cache guard; the native Qwen multimodal batch adapter must be completed and tested on a Gadi GPU allocation before claiming end-to-end training. PBS submission is manual: `qsub pbs/smoke_test.pbs`, then inspect `qstat -f JOBID` and stdout/stderr.
