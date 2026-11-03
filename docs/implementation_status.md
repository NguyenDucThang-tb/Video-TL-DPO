# Trạng thái triển khai

Đã triển khai: schema preference, split theo video, binary TRAIN-only builder, validator target span, uniform frame sampling, temporal/spatiotemporal masking, vanilla DPO, target-generation loss, target-condition video objective, LoRA/reference helpers, metric primitives, config/PBS skeleton và CPU tests.

Chưa xác nhận trong container hiện tại: dataset thật (không có `/scratch`), Qwen checkpoint, CUDA/BF16, forward native video của Transformers, training/evaluation số liệu. Không có kết quả khoa học nào được bịa ra.

Target-condition chỉ nhận record `annotation_verified=true` và evidence hợp lệ. Binary Yes/No là baseline DPO, không phải reproduction đầy đủ của TL-DPO.
