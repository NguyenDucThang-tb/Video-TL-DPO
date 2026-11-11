# Vanilla DPO: chỉ LoRA language attention

Đây là ablation về vị trí adapter của Vanilla DPO, không phải đầy đủ TL-DPO.
Không có điều kiện chọn kết quả theo điểm test hoặc bắt buộc thấp hơn một phương
pháp khác. Báo cáo cả kết quả tăng và giảm.

## Cấu hình

- Chọn `--lora-scope language_attention` để chỉ gắn LoRA vào `q_proj`, `k_proj`,
  `v_proj`, `o_proj` trong các layer attention của language model.
- Projector/merger, vision encoder, language MLP và trọng số base đều frozen.
- Giữ nguyên r16/alpha32/dropout0.05, lr1e-4, beta0.1, một epoch, batch1 × accum8,
  seed0, 300 train records, 8 frames và prompt/scoring của unified v3.
- Mặc định `attention_projector` vẫn là cấu hình đầy đủ. Không đồng thời giảm
  rank/lr vì thí nghiệm này đo riêng tác động của bỏ projector.
- Scope được lưu trong settings của training/evaluation và pipeline manifest;
  checkpoint có scope khác không được resume. Trước optimizer, kiểm tra mọi
  trainable parameter phải là LoRA A/B của language attention; sai thì dừng.
- Run mặc định mới: `outputs/unified_adaptive_seed_v3_language_attention/`.
  Mỗi backend bắt đầu từ pretrained base và LoRA mới; không dùng adapter cũ.

## Chạy trên Gadi

Sau khi push local và pull/rsync code mới, chạy tại login node. Các lệnh bên trong
subshell không làm logout shell ngoài khi lỗi. Không cần build lại archive env.

```bash
if (
  cd /g/data/hn98/$USER/Video-TL-DPO || exit 1
  export VIDEO_TLDPO_CANONICAL_TRAIN="$PWD/artifacts/train_binary.jsonl"
  export VIDEO_TLDPO_EVENT_ROOT="/g/data/hn98/$USER/datasets_video_hallu/eventhallusion_full"
  export VIDEO_TLDPO_EVENT_QUESTIONS="/scratch/jp09/$USER/video-hallucination-benchmark/external/EventHallusion/questions"
  unset VIDEO_TLDPO_OUT VIDEO_TLDPO_MODEL VIDEO_TLDPO_ARCHIVE VIDEO_TLDPO_RUN_ROOT

  bash scripts/submit_unified_pipeline.sh \
    --backends qwen3 internvl35 --lora-scope language_attention \
    --preflight-only &&
  bash scripts/submit_unified_pipeline.sh \
    --backends qwen3 internvl35 --lora-scope language_attention \
    --train-jobs 1 --walltime 00:30:00
); then
  qstat -u "$USER"
else
  echo "Dừng vì lỗi; shell vẫn mở. Kiểm tra qstat và pipeline_jobs.json trước khi chạy lại."
fi
```

Pipeline đầy đủ: 14 GPU jobs và 1 CPU calibration job cho hai model; mỗi GPU job
30 phút. Chúng gồm train, base/adapter VH_VAL, VH_TEST, EH_ALL. Adapter evaluation
chờ training và base tương ứng. Calibration chỉ fit ngưỡng trên VH_VAL; báo cáo
cả raw/calibrated như cấu hình đầy đủ.

Muốn chỉ train (không nộp base/evaluation), dùng:

```bash
if (
  cd /g/data/hn98/$USER/Video-TL-DPO || exit 1
  export VIDEO_TLDPO_CANONICAL_TRAIN="$PWD/artifacts/train_binary.jsonl"
  unset VIDEO_TLDPO_OUT VIDEO_TLDPO_MODEL VIDEO_TLDPO_ARCHIVE VIDEO_TLDPO_RUN_ROOT
  for BACKEND in qwen3 internvl35; do
    bash scripts/submit_unified.sh train "$BACKEND" \
      --lora-scope language_attention --preflight-only || exit 1
  done
  for BACKEND in qwen3 internvl35; do
    bash scripts/submit_unified.sh train "$BACKEND" \
      --lora-scope language_attention --jobs 1 --walltime 00:30:00 || exit 1
  done
); then
  qstat -u "$USER"
else
  echo "Dừng vì lỗi; shell vẫn mở."
fi
```

Đọc kết quả sau khi pipeline hoàn tất:

```bash
cat /g/data/hn98/$USER/Video-TL-DPO/outputs/unified_adaptive_seed_v3_language_attention/results.csv
```

Code có source fingerprint chặt: không đồng bộ code mới vào workspace của một
run cũ còn đang chạy/resume. Dùng code cố định trong toàn bộ training/evaluation/
calibration của run mới; source thay đổi thì contract cũ bị từ chối.

Diagnostic GPU hiện có cũng nhận `--lora-scope language_attention`. Trong một
interactive GPU job đã bung archive và đặt `PYTHONPATH`, chạy
`scripts/smoke_videohallucer_backend.py` với các tham số model/data/output như
trước và thêm tùy chọn này; dùng output mới. Diagnostic kiểm tra gradient,
optimizer update và lưu adapter, không phải kết quả thực nghiệm đầy đủ.

## Kiểm chứng

CPU tests kiểm tra lựa chọn module, loại projector/vision/MLP, guard trainable
parameters, scope trong settings, preflight và chuyển scope sang PBS. Không có
torch/peft/GPU local nên chưa kiểm chứng adapter thực hoặc backward của ablation
trên Qwen3/InternVL. Cần log Gadi có `language_attention_trainables: PASS`, danh
sách `lora_targets`, số trainable parameters và training hoàn tất để xác nhận.
