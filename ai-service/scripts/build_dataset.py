"""BƯỚC A: tạo bộ dữ liệu từ DeepFashion (chưa đụng tới Qdrant).

Đặt file này tại: ai-service/scripts/build_dataset.py

Với mỗi ảnh được chọn: cắt theo bounding box -> CLIP -> vector 512 số (đã chuẩn hóa),
suy ra màu bằng CLIP zero-shot, rồi ghi ra thư mục seed/:
    items.jsonl   mỗi dòng một món đồ (id, đường dẫn ảnh, category, attributes, màu)
    vectors.npy   vector của các món, cùng thứ tự với items.jsonl (float16)
    meta.json     thông tin về lần chạy (model, số lượng, số chiều)

Chỉ cần MỘT người chạy (nên là máy có GPU), rồi push seed/ lên git.
Việc nạp vào Qdrant do scripts/load_qdrant.py làm (bước B).

    python scripts/build_dataset.py --dry-run --limit 2000     # chỉ kiểm tra đọc nhãn
    python scripts/build_dataset.py --limit 500                # chạy thử nhỏ
    python scripts/build_dataset.py --limit 20000              # bản dùng thật
"""
import argparse
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ANNO_SUBDIR = Path("Category and Attribute Prediction Benchmark") / "Anno_coarse"
CATEGORY_TYPES = {1: "upper", 2: "lower", 3: "full"}
# DeepFashion không có nhóm thuộc tính "màu" riêng -> suy ra màu bằng CLIP zero-shot.
COLORS = ["black", "white", "gray", "beige", "brown", "red",
          "pink", "orange", "yellow", "green", "blue", "purple"]


# ----------------------------------------------------------------- đọc nhãn
def data_rows(path):
    """Bỏ 2 dòng đầu (số lượng + tên cột), trả về các dòng dữ liệu."""
    with open(path, encoding="utf-8") as f:
        next(f)
        next(f)
        for line in f:
            line = line.strip()
            if line:
                yield line


def load_categories(anno):
    names, types = [], []
    for line in data_rows(anno / "list_category_cloth.txt"):
        name, ctype = line.rsplit(None, 1)
        names.append(name.strip())
        types.append(CATEGORY_TYPES.get(int(ctype), "unknown"))
    return names, types


def load_attr_names(anno):
    return [line.rsplit(None, 1)[0].strip() for line in data_rows(anno / "list_attr_cloth.txt")]


def load_image_categories(anno, n_categories):
    """Trả về (danh sách tên ảnh theo thứ tự file, danh sách nhãn category 0-based)."""
    images, labels = [], []
    for line in data_rows(anno / "list_category_img.txt"):
        name, label = line.split()
        images.append(name)
        labels.append(int(label))
    offset = 1 if min(labels) >= 1 else 0  # file gốc đánh số từ 1
    labels = [v - offset for v in labels]
    if max(labels) >= n_categories:
        raise ValueError("Nhãn category vượt quá số category, kiểm tra lại file.")
    return images, labels


def load_bboxes(anno):
    boxes = {}
    for line in data_rows(anno / "list_bbox.txt"):
        parts = line.split()
        boxes[parts[0]] = tuple(int(v) for v in parts[1:5])
    return boxes


def load_attributes_for(anno, wanted, attr_names):
    """Đọc dòng-theo-dòng file list_attr_img.txt, chỉ giữ ảnh nằm trong tập được chọn."""
    result = {}
    for line in data_rows(anno / "list_attr_img.txt"):
        name, rest = line.split(None, 1)
        if name in wanted:
            values = rest.split()
            result[name] = [attr_names[i] for i, v in enumerate(values) if v == "1"]
            if len(result) == len(wanted):
                break
    return result


def stratified_sample(images, labels, limit, seed):
    """Chọn đều giữa các category (xoay vòng), tái lập được nhờ seed."""
    if limit <= 0 or limit >= len(images):
        return list(range(len(images)))
    rng = random.Random(seed)
    groups = defaultdict(list)
    for idx, lab in enumerate(labels):
        groups[lab].append(idx)
    for g in groups.values():
        rng.shuffle(g)
    picked, pools = [], list(groups.values())
    while len(picked) < limit and pools:
        for pool in list(pools):
            picked.append(pool.pop())
            if not pool:
                pools.remove(pool)
            if len(picked) >= limit:
                break
    return sorted(picked)


def build_records(data_dir, limit, seed):
    anno = data_dir / ANNO_SUBDIR
    cat_names, cat_types = load_categories(anno)
    attr_names = load_attr_names(anno)
    images, labels = load_image_categories(anno, len(cat_names))
    boxes = load_bboxes(anno)
    chosen = stratified_sample(images, labels, limit, seed)
    attrs = load_attributes_for(anno, {images[i] for i in chosen}, attr_names)
    records = []
    for i in chosen:
        name = images[i]
        records.append({
            "id": i,  # vị trí trong file gốc: ổn định, chạy lại sẽ ghi đè chứ không nhân đôi
            "image": name,
            "bbox": boxes.get(name),
            "payload": {
                "image": name,
                "category": cat_names[labels[i]],
                "category_type": cat_types[labels[i]],
                "attributes": attrs.get(name, []),
            },
        })
    return records, len(images)


# ------------------------------------------------------------- xử lý ảnh
def load_crop(task):
    from PIL import Image
    root, name, bbox = task
    try:
        img = Image.open(root / name).convert("RGB")
    except Exception:  # file thiếu hoặc hỏng
        return None
    if bbox:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        if x2 - x1 > 8 and y2 - y1 > 8:
            img = img.crop((x1, y1, x2, y2))
    return img


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Tạo seed/items.jsonl + seed/vectors.npy từ DeepFashion")
    ap.add_argument("--data-dir", type=Path, default=root / "data")
    ap.add_argument("--out-dir", type=Path, default=root / "seed")
    ap.add_argument("--limit", type=int, default=20000, help="số ảnh (0 = tất cả)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--dry-run", action="store_true", help="chỉ đọc nhãn và in thống kê")
    args = ap.parse_args()

    t0 = time.time()
    records, total = build_records(args.data_dir, args.limit, args.seed)
    print(f"Đã đọc nhãn: chọn {len(records)} / {total} ảnh ({time.time() - t0:.0f}s)")

    if args.dry_run:
        types = Counter(r["payload"]["category_type"] for r in records)
        cats = Counter(r["payload"]["category"] for r in records)
        missing = sum(1 for r in records[:200] if not (args.data_dir / r["image"]).exists())
        no_attr = sum(1 for r in records if not r["payload"]["attributes"])
        print("Theo loại:", dict(types))
        print(f"Số category: {len(cats)} (ít nhất {min(cats.values())}, nhiều nhất {max(cats.values())} ảnh)")
        print(f"Ảnh không có attribute dương tính nào: {no_attr}")
        print(f"File ảnh không tìm thấy trong 200 ảnh đầu: {missing}")
        print("Ví dụ:", records[0]["payload"])
        return

    import json

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app import config
    from app.clip_utils import load_clip

    clip = load_clip(config.CLIP_MODEL, config.CLIP_PRETRAINED, args.device)
    print(f"Model CLIP {config.CLIP_MODEL}/{config.CLIP_PRETRAINED} chạy trên: {clip.device}")
    color_vecs = clip.encode_texts([f"a photo of a {c} clothing item" for c in COLORS])

    items, chunks = [], []
    skipped = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for b in range(0, len(records), args.batch_size):
            batch = records[b:b + args.batch_size]
            tasks = [(args.data_dir, r["image"], r["bbox"]) for r in batch]
            loaded = [(r, im) for r, im in zip(batch, pool.map(load_crop, tasks)) if im is not None]
            skipped += len(batch) - len(loaded)
            if not loaded:
                continue
            vecs = clip.encode_images([im for _, im in loaded])
            color_idx = (vecs @ color_vecs.T).argmax(dim=1).cpu().tolist()
            chunks.append(vecs.cpu().numpy().astype(np.float32))
            for (r, _), ci in zip(loaded, color_idx):
                items.append({"id": r["id"], **r["payload"], "color": COLORS[ci]})
            rate = len(items) / max(time.time() - start, 1e-6)
            eta = (len(records) - len(items) - skipped) / max(rate, 1e-6) / 60
            print(f"{len(items)}/{len(records)} ảnh | {rate:.1f} ảnh/s | còn ~{eta:.0f} phút | bỏ qua {skipped}")

    if not items:
        raise SystemExit("Không xử lý được ảnh nào. Kiểm tra lại --data-dir.")
    vectors = np.concatenate(chunks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    np.save(args.out_dir / "vectors.npy", vectors.astype(np.float16))
    meta = {"model": f"{config.CLIP_MODEL}/{config.CLIP_PRETRAINED}", "count": len(items),
            "dim": int(vectors.shape[1]), "limit": args.limit, "seed": args.seed}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Xong: {len(items)} món (bỏ qua {skipped}) -> {args.out_dir}. Tổng {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
