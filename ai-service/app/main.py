from fastapi import FastAPI
from qdrant_client import QdrantClient

from . import config

app = FastAPI(title="Fashion AI Service")

_qdrant = QdrantClient(url=config.QDRANT_URL)
_clip = {}  # cache lazy: model + processor chỉ tải khi gọi lần đầu


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
    """Hello world CLIP: trả về embedding của một câu mô tả.
    Lần gọi đầu sẽ tải model (~600MB)."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    if "model" not in _clip:
        _clip["model"] = CLIPModel.from_pretrained(config.CLIP_MODEL).eval()
        _clip["proc"] = CLIPProcessor.from_pretrained(config.CLIP_MODEL)

    inputs = _clip["proc"](text=[payload.get("text", "")], return_tensors="pt", padding=True)
    with torch.no_grad():
        vec = _clip["model"].get_text_features(**inputs)
    vec = torch.nn.functional.normalize(vec, dim=-1)[0]
    return {"dim": vec.shape[0], "embedding": vec.tolist()}
