# Data generation và audit

VideoHallucer basic/hallucination là question pair, không phải chosen/rejected của cùng một question. `build_preferences.py` chỉ tạo binary chosen/rejected từ cùng một item TRAIN. Long-form target cần baseline generations, minimally corrected text, character offsets và review evidence; phần chưa review phải để `annotation_verified=false`, không vào main condition experiment.
