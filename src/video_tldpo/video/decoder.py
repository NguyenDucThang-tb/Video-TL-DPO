def decode_video(path, indices, resolution=None):
    try:
        from decord import VideoReader, cpu
    except ImportError as e:
        raise RuntimeError("Cần cài decord trong GPU environment để decode video") from e
    vr = VideoReader(str(path), ctx=cpu(0))
    frames = vr.get_batch(indices).asnumpy()
    if resolution:
        from PIL import Image
        frames = [Image.fromarray(x).convert("RGB").resize((resolution, resolution)) for x in frames]
    return frames, float(len(vr)) / float(vr.get_avg_fps() or 30.0)
