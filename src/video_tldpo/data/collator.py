class PreferenceCollator:
    """Giữ records raw; processor/model backend xử lý video ở training runtime."""
    def __call__(self, batch):
        return {k: [x.get(k) for x in batch] for k in batch[0]}
