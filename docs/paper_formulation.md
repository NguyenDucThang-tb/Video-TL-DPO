# Công thức

Với `x=(video, question)`, vanilla DPO dùng `-log sigmoid(beta * ((log pi(y+|x)-log ref(y+|x))-(log pi(y-|x)-log ref(y-|x))))`.

Target Generation thay tổng trên toàn response bằng tổng tại các token thuộc character spans đã align bằng offset; prompt, padding và phần không liên quan bị mask.

Target Condition giữ nguyên input text và thay video `V` bằng `V_masked`, trong đó chỉ evidence frames/regions đã verified bị corruption. Objective so sánh log-ratio của response đúng ở hai điều kiện. Đây là video extension của image target-condition loss, không phải SFT hay contrastive loss.
