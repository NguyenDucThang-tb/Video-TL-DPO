# Unified v3: cấu hình mới và calibration

Nguồn cấu hình: bản `unified_config (1).md` người dùng cung cấp; bản nguyên văn
nằm trong `UNIFIED_CONFIG_REFERENCE.md`.

Protocol: `unified_adaptive_seed_v3`. Train vẫn là Vanilla DPO trên token đáp án,
beta0.1, LoRA r16/alpha32/dropout0.05 trên attention+projector, lr1e-4, batch1 ×
accumulation8, 300 câu/38 optimizer steps, seed0, một epoch. Mỗi run mới bắt đầu
từ pretrained base + LoRA mới. Resume chỉ tiếp tục checkpoint cùng contract.
Target Generation/Condition của full TL-DPO chưa được tích hợp vào runner này.

Không thay split: 150 train pairs/50 val pairs/155 test pairs. Dùng split.json và
eval_vh.jsonl chung để đối chiếu membership. Theo bản mới, ưu tiên canonical
`OPA-DPO-Video/outputs/work_binary_all/sft.jsonl` nếu file có thật. Trên Gadi file
này trước đó không tồn tại; fallback là train_binary.jsonl riêng của Video-TL-DPO.
Cả hai nguồn đều phải chứa đúng 300 train sample IDs và cùng question/label/video
với artifact đã chốt. Có thể cố định nguồn riêng bằng VIDEO_TLDPO_CANONICAL_TRAIN;
preflight in đường dẫn và SHA256 thực dùng. Không gọi build SFT của thí nghiệm khác.

Video/prompt/margin vẫn theo bản cấu hình: 8 frame đều, RGB 448 bicubic, fps
8/(N/avg_fps), single forward cuối prompt, Yes/No chữ hoa, margin>0 dự đoán yes.
Base VH_TEST/EH_ALL phải khớp expected counts trước khi xuất bảng cuối.

## Ngưỡng theo VH_VAL

Mỗi base và adapter được chấm đủ VH_VAL100, VH_TEST310 và EH_ALL409. Các job
độc lập lưu margins gốc. Calibration đọc file đã hoàn tất, kiểm tra source,
model/runtime/data hashes và input digests base/adapter.

- Candidate t: 0 và midpoint giữa các margin VH_VAL liên tiếp sau khi sort.
- Chọn q_acc VH_VAL cao nhất, tie lấy |t| nhỏ nhất. Nếu cả |t| cũng bằng nhau,
  chọn t âm trước làm tie-break xác định; không dùng test để phá tie.
- Mỗi variant có một t riêng, dùng nguyên t đó cho VH_TEST và EH_ALL.
- Quy tắc strict: yes khi margin>t, bằng t là no.
- Không sửa raw margins. Báo cáo cả raw và calibrated, kể cả khi điểm giảm.

## Toàn bộ batch pipeline

`submit_unified_pipeline.sh` mặc định nộp cho Qwen3 và InternVL3.5, GPU walltime
30 phút/job, env archive được bung vào PBS_JOBFS. Các đường dẫn lớn ở GData;
mount scratch chỉ để đọc dataset/canonical EventHallusion.

Mỗi backend với --train-jobs1 có bảy GPU jobs: train1, base3 (VH_VAL/VH_TEST/EH_ALL),
adapter3. Mỗi adapter job chờ afterok train và base tương ứng. Một CPU job chung
chờ cả sáu adapter evaluations rồi xuất calibration. Tổng hai model: 14 GPU jobs
và một CPU job. Với --train-jobs2, tổng là 16 GPU jobs + một CPU job; job train
thứ hai resume cùng run và bỏ qua optimizer nếu run đã complete.

Train không phải đợi base. Khi báo cáo, base phải đã PASS. Mọi job dependencies
dùng afterok, không dùng array. Job IDs được lưu ngay sau mỗi lần qsub thành công
để theo dõi nếu submit bị gián đoạn. Không tự retry qsub khi timeout.

```bash
cd /g/data/hn98/$USER/Video-TL-DPO
export VIDEO_TLDPO_CANONICAL_TRAIN="$PWD/artifacts/train_binary.jsonl"
export VIDEO_TLDPO_EVENT_ROOT=/g/data/hn98/$USER/datasets_video_hallu/eventhallusion_full
export VIDEO_TLDPO_EVENT_QUESTIONS=/scratch/jp09/$USER/video-hallucination-benchmark/external/EventHallusion/questions
unset VIDEO_TLDPO_OUT VIDEO_TLDPO_MODEL VIDEO_TLDPO_ARCHIVE

bash scripts/submit_unified_pipeline.sh --backends qwen3 internvl35 --preflight-only
bash scripts/submit_unified_pipeline.sh --backends qwen3 internvl35 --train-jobs 1 --walltime 00:30:00
```

Các đường dẫn EH trên lấy từ log thực tế người dùng đã cung cấp. Nếu thư mục đã
di chuyển tiếp, đổi hai env variables trước preflight. Không tải dataset/model/env.

Run mặc định dùng outputs/unified_adaptive_seed_v3. Run v2 giữ nguyên. Để chạy một
thí nghiệm mới khác, dùng --run-name unified_adaptive_seed_v3_repeat1 ở cả preflight
và submit. Không tái sử dụng tên run để train từ đầu. --resume chỉ dùng sau khi
các job cũ đều dừng để tiếp tục run đang dở. Công cụ kiểm tra job history và từ
chối nếu không xác nhận được trạng thái cũ.

Nếu evaluation trả PARTIAL/code3, calibration không chạy thành công. Chờ các job
dừng rồi resubmit run với --resume; saved margins và same-run checkpoint được
tiếp tục. Không xem qsub acceptance hoặc COMPLETED của một stage là toàn bộ run.

## Đọc kết quả

```bash
ROOT=/g/data/hn98/$USER/Video-TL-DPO/outputs/unified_adaptive_seed_v3
cat "$ROOT/results.csv"
cat "$ROOT/qwen3/calibration_report.json"
cat "$ROOT/internvl35/calibration_report.json"
```

results.csv có bốn dòng mỗi backbone: base raw, base calibrated, adapter raw,
adapter calibrated, gồm VH_Q_acc/VH_pair_acc/EH_acc/yes_rate/t. calibration_report
có ba chênh lệch theo mục9. Mỗi variant có calibration/<variant>/calibrated.csv,
threshold.json và margins.jsonl tương thích id/margin chung (819 val+VHtest+EH).
Các file margins inference gốc nằm riêng trong dataset_split/<variant>.

Chỉ calibration hoặc xuất lại bảng từ margins hoàn tất, không dùng GPU:

```bash
python3 scripts/calibrate_unified.py --backends qwen3 internvl35
```

## Phạm vi kiểm chứng

CPU tests kiểm tra candidate/tie, strict threshold, không chọn t từ test,
đối chiếu contracts và hashes, giữ nguyên margins, export cùng t cho cả hai test,
và graph/dependency PBS. Các lời gọi PBS trong unit test là mô phỏng, không phải
job Gadi thật. Chưa xác nhận runtime của protocol v3 trên Gadi cho đến khi có log.
