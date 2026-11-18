import os

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "fashion_items")

# Model CLIP (thư viện open_clip). ViT-B-32 + trọng số "openai" cho vector 512 chiều.
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "openai")

# auto = tự chọn cuda -> mps -> cpu. Ép thủ công: DEVICE=cpu | cuda | mps
DEVICE = os.getenv("DEVICE", "auto")