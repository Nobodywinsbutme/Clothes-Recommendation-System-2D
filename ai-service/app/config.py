import os

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
COLLECTION = os.getenv("QDRANT_COLLECTION", "fashion_items")
