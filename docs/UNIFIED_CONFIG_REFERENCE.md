# Cấu hình thống nhất cho Adaptive-SEED, OPA-DPO-Video và TL-DPO

Mọi phương pháp (SFT, DPO, OPA-DPO, TL-DPO, SEED) dùng **cùng một** quy trình eval (mục 1–5) và
**cùng một** budget train (mục 7). Nhờ vậy chênh lệch giữa các dòng trong bảng chỉ đến từ phương pháp.
Quy trình này là của Adaptive-SEED (`src/seed_video/models/*.py`, `evaluation/`); OPA-DPO-Video
(`opadpo_video/evaluate.py`) tái hiện nó.

**Điều kiện bắt buộc:** trước khi báo cáo, chạy mô hình gốc (không train) bằng pipeline của phương pháp
và kiểm tra base ra **đúng** các số ở mục 6. Nếu lệch thì pipeline eval đang khác, và kết quả sau train
không so sánh được với các phương pháp khác.

Bảng kết quả gồm một dòng **Base** (mô hình gốc, không train), rồi mỗi dòng sau là mô hình gốc + một
phương pháp, chấm trên cùng 3 cột: VH Q acc, VH pair acc, EH acc.

## 1. Dữ liệu

### VideoHallucer
- Nguồn: HF dataset `bigai-nlco/VideoHallucer`, revision `c93475bf1be466c1af90d005749ccb4e6b3ad066`.
- Subset: `temporal` và `object_relation`.
- Mỗi cặp gồm câu `basic` (đáp án "yes") và câu `hallucination` (đáp án "no"). Khử trùng theo
  (video, câu hỏi) của cả hai câu.
- Cách chia: sắp xếp `pair_id`, xáo bằng `random.Random(0)`, rồi lấy lần lượt 150 cặp train, 50 cặp val
  và tối đa 300 cặp test. Bỏ cặp nào có video đã nằm ở split trước, để train, val và test không chung
  video. Kết quả: **train 150 cặp, val 50 cặp (100 câu), test 155 cặp (310 câu)**.
- Dùng thẳng các file có sẵn trên Gadi, không tự chia lại:
  - `/g/data/hn98/dd9648/OPA-DPO-Video/outputs/data/split.json`: pair id của từng split.
  - `/g/data/hn98/dd9648/OPA-DPO-Video/outputs/data/eval_vh.jsonl`: các câu val và test.
  - `/g/data/hn98/dd9648/OPA-DPO-Video/outputs/work_binary_all/sft.jsonl`: các câu train kèm nhãn
    (tạo bằng `scripts/gadi/prepare_binary.sh`).
- Video: `/scratch/jp09/dd9648/datasets_video_hallu/videohallucer`.

### EventHallusion
- Chỉ dùng để eval, không train. Gồm cả **409 câu**: `entire` 114, `misleading` 102, `mix` 193.
- Thư mục: `/scratch/hn98/dd9648/runs/exp1/EventHallusion` (`questions/<split>_questions.json` + video).

## 2. Video

- Đọc bằng decord, lấy **8 frame** đều theo thời gian: `idx = np.linspace(0, N - 1, 8).round()`, với N là
  tổng số frame của video.
- Mỗi frame: PIL RGB, resize về **448 × 448** bicubic, không giữ tỉ lệ.
- fps truyền cho model = `8 / (N / avg_fps)`.
- Theo backbone:
  - Qwen2.5-VL: `processor(text=..., videos=[video], fps=[fps])`.
  - Qwen3-VL: `processor(..., video_metadata=[{"total_num_frames": 8, "fps": fps,
    "duration": 8 / fps, "frames_indices": [0..7]}], do_sample_frames=False)`.
  - InternVL3.5: mỗi frame là 1 ảnh 448 (không chia tile), chuẩn hoá ImageNet
    (mean 0.485/0.456/0.406, std 0.229/0.224/0.225), 256 token ảnh mỗi frame, `image_flags` = 1.

## 3. Prompt

- Nội dung user: `question.strip() + "\nAnswer with yes or no."`
- Qwen: chat template mặc định của processor, video đặt trước text, `add_generation_prompt=True`, không
  thêm system prompt.
- InternVL: conversation template của model (giữ system message mặc định). Nội dung user là
  `"Frame1: <image>\n...Frame8: <image>\n" + câu hỏi`; mỗi `<image>` được thay bằng
  `<img>` + `<IMG_CONTEXT>` × 256 + `</img>`.

## 4. Cách chấm

- Không sinh câu trả lời. Chỉ chạy **một lượt forward** và lấy logit ở vị trí cuối của prompt.
- `margin = logit("Yes") − logit("No")`; dự đoán "yes" khi `margin > 0`.
- Token đáp án là **`Yes` / `No` cho cả 3 backbone**, lấy bằng
  `tokenizer.encode("Yes", add_special_tokens=False)[0]` (tương tự với `"No"`).
- Lưu ý với Qwen3-VL: model này tự trả lời chữ thường ("yes"/"no", khoảng 99% xác suất; "Yes"/"No" dưới
  0.5%). Margin trên "Yes"/"No" vì vậy có thể khác câu model sẽ sinh ra. Quy ước này được giữ để khớp
  các kết quả Adaptive-SEED Qwen3 đã báo cáo. Mọi phương pháp phải dùng **đúng quy ước này**, kể cả
  nhãn train ("Yes."/"No.").
- Batch 1, bf16, `model.eval()`.
- Attention: Qwen dùng `sdpa`; InternVL dùng eager (`use_flash_attn=False`, `trust_remote_code=True`).
- Môi trường: Qwen3-VL và InternVL3.5 cần transformers 4.57.6 (venv `/scratch/jp09/dd9648/venv_qwen3`).

## 5. Chỉ số báo cáo

| Cột | Cách tính |
|---|---|
| VH Q acc | tỉ lệ câu đúng trên 310 câu VH_TEST |
| VH pair acc | tỉ lệ cặp đúng **cả hai** câu trên 155 cặp VH_TEST |
| EH acc | tỉ lệ câu đúng trên 409 câu EventHallusion |

Nên ghi thêm `yes_rate` (tỉ lệ trả lời "yes"). Chỉ dùng VH_VAL (100 câu) để chọn cấu hình, không dùng
test để chọn.

## 6. Base chuẩn (mô hình gốc, không train)

Pipeline nào cũng phải ra đúng các số này:

| Backbone | VH Q acc | VH pair acc | EH acc | Số câu đúng (VH / cặp / EH) |
|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 0.7419 | 0.5032 | 0.6601 | 230/310 · 78/155 · 270/409 |
| Qwen3-VL-8B-Instruct | 0.7581 | 0.5742 | 0.6455 | 235/310 · 89/155 · 264/409 |
| InternVL3_5-8B-Instruct | 0.6839 | 0.4258 | 0.4841 | 212/310 · 66/155 · 198/409 |

Margin base của từng câu nằm ở
`/g/data/hn98/dd9648/OPA-DPO-Video/outputs/eval/{base,base_qwen3,base_internvl}/margins.jsonl`.
Nếu tổng số không khớp, so margin từng câu với file này để tìm chỗ lệch.

## 7. Budget train (student budget của Adaptive-SEED)

| Mục | Giá trị |
|---|---|
| Dữ liệu train | toàn bộ TRAIN split: 150 cặp, khoảng 300 câu, cả `temporal` lẫn `object_relation` |
| Adapter | LoRA r = 16, α = 32, dropout = 0.05, bias none |
| Module có LoRA | attention q/k/v/o của LM + projector thị giác (regex bên dưới) |
| Optimizer | AdamW, lr 1e-4, weight decay 0, clip grad norm 1.0 |
| Lịch lr | cosine, warmup 5% số bước |
| Epoch / batch | 1 epoch, batch 1 × gradient accumulation 8 (batch hiệu dụng 8) |
| Độ chính xác | bf16 |
| Seed | 0 |
| Checkpoint | checkpoint cuối epoch, không early stopping |

Regex module có LoRA (dùng làm `target_modules` của PEFT):

```text
Qwen2.5-VL : (.*\.layers\.\d+\.self_attn\.(q|k|v|o)_proj)|(.*visual\.merger\.mlp\.(0|2))
Qwen3-VL   : (.*\.layers\.\d+\.self_attn\.(q|k|v|o)_proj)|(.*visual\.merger\.(mlp\.(0|2)|linear_fc[12]))
InternVL3.5: (.*language_model\.model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj)|(.*mlp1\.(1|3))
```

Số tham số train được: 10,395,648 với Qwen2.5-VL; 15,597,568 với InternVL3.5.

Hàm loss do từng phương pháp quy định, và phải ghi rõ trong báo cáo, cùng β và các hệ số riêng. Với
phương pháp dạng DPO, reference là mô hình gốc. Các baseline yes/no của OPA-DPO-Video (SFT yes/no,
DPO yes/no, β = 0.1) chỉ tính loss trên token đáp án, đúng token được chấm ở mục 4.

## 8. Chấm checkpoint bằng script chung

Nếu checkpoint là adapter PEFT LoRA trên đúng model gốc ở mục 6, có thể chấm thẳng bằng script của
OPA-DPO-Video. Cách này chắc chắn dùng cùng quy trình eval:

```bash
cd /scratch/hn98/dd9648/OPA-DPO-Video && git pull
qsub -l walltime=00:30:00,mem=64GB,jobfs=10GB \
  -v MODEL=qwen3,NAME=tldpo_qwen3,ADAPTER=/đường/dẫn/adapter,ENV_NAME=none,VENV=/scratch/jp09/dd9648/venv_qwen3 \
  scripts/gadi/eval.pbs
```

`MODEL` là `qwen25`, `qwen3` hoặc `internvl`. Với `qwen25` thì bỏ `ENV_NAME` và `VENV`.
Kết quả nằm ở `/g/data/hn98/dd9648/OPA-DPO-Video/outputs/eval/<NAME>/results.csv`, gồm cả
`wrong_to_correct` và `correct_to_wrong` so với base của cùng backbone.
