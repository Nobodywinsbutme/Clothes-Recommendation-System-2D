"""BƯỚC B: nạp seed/ vào Qdrant (không cần GPU, torch, CLIP hay ảnh).

Đặt file này tại: ai-service/scripts/load_qdrant.py

Đọc seed/items.jsonl + seed/vectors.npy (do build_dataset.py tạo ra) và ghi vào
collection `fashion_items` trong Qdrant. Mọi thành viên chạy bước này sau khi git pull.

    python scripts/load_qdrant.py --recreate
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Nạp seed/ vào Qdrant")
    ap.add_argument("--seed-dir", type=Path, default=root / "seed")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", default="fashion_items")
    ap.add_argument("--recreate", action="store_true", help="xóa collection cũ rồi tạo lại")
    ap.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args()

    import numpy as np
    from qdrant_client import QdrantClient, models

    items_path, vec_path = args.seed_dir / "items.jsonl", args.seed_dir / "vectors.npy"
    if not items_path.exists() or not vec_path.exists():
        raise SystemExit(f"Không thấy items.jsonl / vectors.npy trong {args.seed_dir}. "
                         "Đã git pull chưa, hoặc đã chạy build_dataset.py chưa?")

    vectors = np.load(vec_path).astype(np.float32)
    with open(items_path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if len(items) != len(vectors):
        raise SystemExit(f"Lệch số lượng: {len(items)} món nhưng {len(vectors)} vector.")
    print(f"Đọc {len(items)} món, vector {vectors.shape[1]} chiều.")

    client = QdrantClient(url=args.qdrant_url, check_compatibility=False)
    if args.recreate and client.collection_exists(args.collection):
        client.delete_collection(args.collection)
    if not client.collection_exists(args.collection):
        client.create_collection(
            args.collection,
            vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
        )
        for field in ("category", "category_type", "color"):
            client.create_payload_index(args.collection, field, models.PayloadSchemaType.KEYWORD)

    for b in range(0, len(items), args.batch_size):
        points = [
            models.PointStruct(
                id=it["id"],
                vector=vectors[b + i].tolist(),
                payload={k: v for k, v in it.items() if k != "id"},
            )
            for i, it in enumerate(items[b:b + args.batch_size])
        ]
        client.upsert(args.collection, points=points, wait=True)
        print(f"{min(b + args.batch_size, len(items))}/{len(items)}")
    print(f"Xong: '{args.collection}' có {len(items)} món.")


if __name__ == "__main__":
    main()
