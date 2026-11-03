import numpy as np

def uniform_indices(total_frames, num_frames):
    if total_frames < 1 or num_frames < 1:
        raise ValueError("total_frames và num_frames phải dương")
    return np.linspace(0, total_frames - 1, num_frames).round().astype(int).tolist()

def frame_times(indices, fps):
    if fps <= 0: raise ValueError("fps phải dương")
    return [float(i) / fps for i in indices]
