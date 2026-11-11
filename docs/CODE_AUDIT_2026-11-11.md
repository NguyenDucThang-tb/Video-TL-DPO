# Code audit: điểm VideoHallucer tăng mạnh

Ngày rà soát: 2026-11-11. Commit gốc: `51e3276`.

Ghi chú cập nhật sau audit: lần setup unified v3 theo tài liệu mới đã sửa hai lỗi
indentation test, gọi guard train/EH overlap trong runner, mở rộng source
fingerprint và thêm calibration VH_VAL. Nội dung bên dưới mô tả bản v2 tại thời
điểm audit, không phải danh sách lỗi chưa sửa của v3. Auditor standalone v2 giữ
nguyên để kiểm tra artifact lịch sử.

## Kết luận và phạm vi bằng chứng

Chưa tìm thấy lỗi trong đường chạy unified khiến model nhận nhãn test, cộng điểm sai,
nạp adapter vào base, hoặc đưa cả train/test vào optimizer. Không kết luận vì vậy
đã loại trừ mọi lỗi runtime: máy local không có torch, GPU, model weights hoặc
artifact thực của Gadi. Các điểm số và PBS exit status là log do người dùng cung cấp.

Source fingerprint local là
`4df6df2282884d73e808abaff52f31381e290a2af94982b8a2c462da886c6f05`,
khớp với `source_sha256` trong cả hai train summary của Gadi. Điều này xác nhận
đã rà đúng source được fingerprint trong run; danh sách fingerprint chưa bao phủ
mọi helper và không thay thế kiểm chứng nội dung weights.

Đã quét cú pháp toàn bộ 61 file Python gốc, kiểm tra cú pháp shell/PBS, rà sâu đường
`submit_unified.sh -> run_unified.pbs -> run_unified.py -> data/models/training/evaluation.unified`,
checkpoint helper và các đường legacy/data builder/loss còn lại. Đã đối chiếu
preprocessing và margin với source Adaptive-SEED local. Chưa chạy model thật lại.

## Vì sao điểm có thể tăng nhiều

Loss đang chạy chỉ dùng token `Yes` và `No` ở cuối prompt. Với `s=+1` cho nhãn yes,
`s=-1` cho nhãn no, loss tương đương:

`-log sigmoid(beta * s * (policy_margin - reference_margin))`.

Log-normalizer của full vocabulary triệt tiêu. Đây là binary answer-token DPO hợp
lệ theo cấu hình thống nhất. Nó có thể sửa mạnh thiên lệch Yes/No, chưa chứng minh
model dùng bằng chứng video tốt hơn. Đây là giả thuyết cần kiểm tra bằng control,
không phải kết luận rằng toàn bộ gain chỉ đến từ bias.

VideoHallucer test có 155 nhãn yes và 155 nhãn no. Từ số correct và yes_rate đã gửi,
có thể suy ra chính xác các số sau; script audit sẽ tính lại trực tiếp từ margins:

| Model | Basic đúng (yes) | Hallucination đúng (no) | Yes rate | Cặp đều sai |
|---|---:|---:|---:|---:|
| Qwen3 base | 136/155 = 87.74% | 99/155 = 63.87% | 61.94% | 9 |
| Qwen3 adapter | 129/155 = 83.23% | 134/155 = 86.45% | 48.39% | 5 |
| InternVL3.5 base | 138/155 = 89.03% | 74/155 = 47.74% | 70.65% | 9 |
| InternVL3.5 adapter | 126/155 = 81.29% | 123/155 = 79.35% | 50.97% | 16 |

Gain tập trung vào nhãn no; độ đúng trên nhãn yes giảm. InternVL có nhiều cặp đúng
cả hai câu hơn nhưng số cặp sai cả hai câu cũng tăng. Báo cáo chỉ overall/pair acc
sẽ che mất đánh đổi này.

Bảng cũ người dùng đưa có r8/alpha16/lr1e-5, khác r16/alpha32/lr1e-4 của run mới.
Base cũ cũng khác base mới, nên không so trực tiếp hai mức gain. Đường legacy có
generation/parsing, còn unified dùng margin chữ hoa. Cần run metadata cũ để xác
nhận bảng cũ dùng đường nào. Shared config và source Adaptive-SEED lưu ý Qwen3
có thể sinh chữ thường trong khi metric chấm chữ hoa; tỷ lệ xác suất thực tế của
run hiện tại chưa được đo độc lập tại local.

## Findings

### F1 — Phương pháp thực chạy chưa phải full TL-DPO (ảnh hưởng cách báo cáo)

`src/video_tldpo/training/unified.py:97` gọi binary DPO trực tiếp. Không gọi
Target Generation/Target Condition, không dùng target spans hay masked evidence.
`scripts/train.py` vẫn là scaffold chỉ in READY/TODO_RUNTIME. Có hàm loss target
ở `src/video_tldpo/losses`, nhưng chưa có runtime target training hoàn chỉnh hoặc
char-span -> tokenizer-span pipeline. Vì vậy các số hiện tại phải ghi Vanilla
DPO LoRA theo unified protocol. Không được ghi là kết quả phương pháp full TL-DPO.

### F2 — Base PASS chỉ bảo đảm tổng số, chưa bảo đảm tái hiện từng mẫu (P1)

`src/video_tldpo/evaluation/unified.py:57`: `passed` chỉ dựa trên số câu/cặp đúng.
`prediction_disagreements` với reference được ghi diagnostic nhưng không làm FAIL.
Reference có thể thiếu. Hai pipeline có thể cùng accuracy/pair accuracy nhưng sai
khác mẫu. Kiểm thử hiện tại cố ý cho phép tình huống này.

Điều này làm yếu khẳng định tương đương với pipeline repo khác. Nó không tự tăng
metric và không thay đổi margin. Trong cùng repo, `compare()` vẫn kiểm tra cùng
source/settings/model/data/runtime và `input_digest` từng mẫu giữa base/adapter.

### F3 — Hai file regression test hiện không import được (P1 cho kiểm chứng)

- `tests/test_unified_stdlib.py:65`: unexpected indent.
- `tests/test_unified_resume_cpu.py:64`: unexpected indent (dòng contract trước đó lệch cấp).

Đã tái hiện lỗi bằng AST toàn repo và chạy unittest trên file gốc. Full suite
hiện không thể collection sạch. Không được dùng tuyên bố test PASS cũ để bảo đảm
phiên bản hiện tại. Một bản sửa indentation chỉ trong bộ nhớ đã cho 10 stdlib
tests chạy PASS; file gốc không bị sửa. CPU resume/torch test chưa chạy vì local
không có torch/numpy/safetensors.

### F4 — CLI chuẩn bị dữ liệu legacy bị hỏng và thiếu guard split (P1 nếu dùng)

`data/videohallucer.py` không sinh sample_id, video path thiếu `videos/`, pair-ID
schema khác artifact đang dùng. `cli.py` chỉ giữ basic rồi truyền vào
`preference_builder.binary_record()`, hàm này đòi sample_id và luôn gắn split=train.
Nếu chỉ sửa KeyError mà không sửa split handling, có nguy cơ đánh dấu mọi dữ liệu
đầu vào thành train. Validator chung cũng chưa kiểm tra chosen/rejected/GT đầy đủ.

Đường này không được runner unified dùng. Hash train của run hiện tại là
`c27370a5e27d231fa8fe54b99788a63254ae58075c6acb26c13af1f2dc23ad8d`;
preflight unified kiểm tra sample/pair membership và video separation trước train.
Chưa thấy bằng chứng F4 đã gây leakage trong các run hiện tại.

### F5 — Unified EH thiếu guard train/EH overlap và byte-duplicate check (P2)

Nhánh EH trong `scripts/run_unified.py:74` chỉ gọi `load_eventhallusion()`; không
gọi helper `validate_eventhallusion_inputs()` có kiểm tra train-video-path overlap.
VideoHallucer vẫn có guard train/val/test trong `data/unified.py`.

Người dùng đã chạy audit path/name trên Gadi: train-val, train-test, train-EH,
val-test đều không trùng; train có 300 unique IDs. Chưa kiểm tra video copies đổi
tên hoặc re-encoding. Auditor mới có tùy chọn SHA256 cho byte-identical copies.

### F6 — Provenance chưa kiểm tra đầy đủ contents (P2)

`models/unified.py:identity()` ghi sizes của weight shards, không hash weight bytes
và không đối chiếu nội dung weights với manifest chính thức. `source_fingerprint()`
bỏ qua `evaluation/eventhallusion.py` và helper `evaluation/video_validation.py`.
Dataset revision trong SETTINGS là khai báo; artifact không tự chứng minh source
video đúng revision đó. Đây là khoảng trống provenance, không phải bằng chứng
weights/data hiện tại bị sai.

### F7 — Các điểm nhỏ và legacy

- `answer_ids()` dùng token đầu tiên, chưa yêu cầu Yes/No mỗi từ đúng một token;
  cần in tokenizer IDs thực tế tại Gadi để xác nhận.
- `loss_first` và `loss_last` là hai mẫu khác nhau trong thứ tự shuffle. Riêng
  `loss_last=2.8` của Qwen3 không đủ kết luận cả training bị xấu đi; cần loss mean
  từng accumulation group. Auditor mới xuất đầy đủ 38 group means.
- Checkpoint helper dùng biến `state` còn lại từ vòng quét thay vì `metadata`
  trong một kiểm tra protocol. Các candidate hiện đã bắt buộc cùng contract nên
  chưa thấy tác động tới các run đã báo cáo.
- PBS dùng deadline cố định 2700 giây. Nếu qalter walltime xuống 1800 giây thì
  deadline nội bộ không tự đổi; có thể bị PBS cắt trước khi pause chủ động.
- `setup_vlm8b_assets.pbs` vẫn dừng ở `pip check` khi gặp warning metadata decord
  đã báo trước; env hiện tại được người dùng đóng gói thủ công sau bước sửa.
- Test `test_target_generation_ignores_unrelated_tokens` không tính lại loss sau
  khi đổi tensor nên chưa kiểm chứng điều tên test mô tả. Các loss tests là CPU
  checks, không xác nhận full target training.

## Những phần đã truy vết và không thấy lỗi ở active runner

- Forward chỉ nhận video, question, prompt template; không nhận gt_answer/pair_id/split.
- Logit cuối prompt dự đoán token đầu response: vị trí causal đúng cho answer-only.
- Scores chuyển float32 trước log_softmax và trừ margin; DPO sign đúng cho cả yes/no.
- Reference load riêng, eval và requires_grad=False, dùng no_grad; không optimizer reference.
- LoRA scope attention+projector; optimizer chỉ lấy parameters requires_grad.
- 300 records, shuffle seed0, accumulation8, 38 optimizer steps, 1 epoch.
  Nhóm cuối4 chia8 là quy ước đã ghi trong cấu hình.
- Checkpoint lưu sample order/cursor, loss count, optimizer/scheduler/RNG; run đã
  completed được job sau nhận diện để không train lại từ record0.
- Prediction là margin>0; correctness so với label, pair acc đòi cả basic/hallucination
  đúng. Không thấy sử dụng expected base counts để sửa margin hoặc sửa dự đoán.
- Eval metadata chứa adapter SHA và từ chối resume nếu checkpoint khác. Base chỉ
  nạp model gốc; adapter variant mới gọi PeftModel.from_pretrained.

## Kiểm chứng local và bước Gadi còn lại

Đã thêm `scripts/audit_unified_run.py` (stdlib, không GPU/model download) và bốn
regression tests. Bốn test mới PASS, gồm metric đối chiếu số đếm tay, tình huống
chèn test ID vào checkpoint train order, giả correct flag và copy video đổi tên. Script mới chưa chạy với
artifact Gadi. Nó không sửa model, split, checkpoint hoặc predictions; chỉ đọc và
ghi một JSON report. Cần sync script lên Gadi trước khi chạy:

```bash
PROJECT=/g/data/hn98/$USER/Video-TL-DPO
python3 "$PROJECT/scripts/audit_unified_run.py" --project "$PROJECT"
```

Để kiểm tra byte-identical video copies, chạy cùng lệnh thêm `--hash-videos` trong
compute/copyq job có mount gdata/hn98+scratch/hn98+scratch/jp09. Chỉ hash các size
có mặt ở nhiều group để tránh đọc toàn bộ video không cần thiết. SHA256 không
phát hiện cùng cảnh bị re-encode hoặc crop.

Control cần thiết để giải thích gain: thu base margins của VH validation (100 câu),
chọn một constant threshold bằng validation, cố định threshold rồi chấm test đã
lưu. Không chọn threshold trên test. Nếu control calibration đạt gần adapter,
gain chưa thể quy hoàn toàn cho visual grounding. Có thể chấm thêm uppercase,
lowercase và generation như các diagnostic đã định trước, giữ metric chính của
shared protocol và báo cáo toàn bộ kết quả.

Không sửa core training/evaluation trong lần audit này, không push và không submit job.
