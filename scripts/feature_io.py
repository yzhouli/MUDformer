from pathlib import Path
import numpy as np
from mudformer.data import ManifestRecord

def load_features(record: ManifestRecord, feature_root: Path, max_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(feature_root / record.path, allow_pickle=False) as data:
        text = np.asarray(data["text"], dtype=np.float32)
        image = np.asarray(data["image"], dtype=np.float32)
        image_present = np.asarray(data["image_present"], dtype=np.bool_)
    if text.ndim != 2 or image.shape != text.shape or image_present.shape != (text.shape[0],):
        raise ValueError(f"Invalid cached features: {feature_root / record.path}")
    if not text.shape[0]:
        raise ValueError(f"Empty cached feature sequence: {feature_root / record.path}")
    start = max(0, text.shape[0] - max_length)
    return text[start:].copy(), image[start:].copy(), image_present[start:].copy()
