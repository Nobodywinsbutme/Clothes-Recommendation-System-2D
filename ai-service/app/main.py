from fastapi import FastAPI
from qdrant_client import QdrantClient

from . import config

app = FastAPI(title="Fashion AI Service")

# check_compatibility=False: tắt cảnh báo lệch phiên bản giữa qdrant-client và server Docker.
_qdrant = QdrantClient(url=config.QDRANT_URL, check_compatibility=False)
_clip = {}  # cache lazy: model chỉ tải khi gọi lần đầu


def get_clip():
    if "clip" not in _clip:
        from .clip_utils import load_clip

        _clip["clip"] = load_clip(config.CLIP_MODEL, config.CLIP_PRETRAINED, config.DEVICE)
    return _clip["clip"]


@app.get("/health")
def health():
    return {"service": "ai-service", "status": "ok"}


@app.get("/health/qdrant")
def health_qdrant():
    try:
        names = [c.name for c in _qdrant.get_collections().collections]
        return {"qdrant": "ok", "collections": names}
    except Exception as e:  # noqa: BLE001
        return {"qdrant": "unreachable", "error": str(e)}


@app.post("/embed/text")
def embed_text(payload: dict):
    """Trả về embedding CLIP của một câu mô tả. Lần gọi đầu sẽ tải model."""
    clip = get_clip()
    vec = clip.encode_texts([payload.get("text", "")])[0]
    return {"dim": int(vec.shape[0]), "device": clip.device, "embedding": vec.tolist()}