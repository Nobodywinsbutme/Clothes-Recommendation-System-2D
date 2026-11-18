"""Normalized experiment input. No fabricated user histories or outfit labels."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .baseline import slot_for_category


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return parsed


def load_catalog(seed_dir):
    # NumPy is an existing application dependency. It is not needed by smoke tests.
    import numpy as np

    root = Path(seed_dir)
    items = [json.loads(line) for line in (root / "items.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    vectors = torch.from_numpy(np.load(root / "vectors.npy", allow_pickle=False).astype("float32"))
    ids = [str(item["id"]) for item in items]
    if len(ids) != len(set(ids)) or vectors.shape != (len(items), meta["dim"]):
        raise ValueError("Catalog IDs/count/dimension are inconsistent")
    if meta["count"] != len(items) or not torch.isfinite(vectors).all():
        raise ValueError("Catalog metadata or vector values are invalid")
    if (vectors.norm(dim=-1) <= 1e-8).any():
        raise ValueError("Catalog contains zero vectors")
    identity = hashlib.sha256()
    for name in ("meta.json", "items.jsonl", "vectors.npy"):
        with (root / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                identity.update(block)
    return {"items": items, "ids": ids, "index": {key: i for i, key in enumerate(ids)},
            "vectors": F.normalize(vectors, dim=-1), "meta": meta,
            "sha256": identity.hexdigest(),
            "slots": [slot_for_category(item["category"]) for item in items]}


def validate_samples(samples, catalog, *, require_all_splits=True):
    """Reject missing IDs, future history and duplicate outfits across splits.

    Upstream must additionally group near-duplicate photos/products. This function
    cannot verify whether claimed labels/timestamps are genuine.
    """
    if not samples:
        raise ValueError("No samples provided")
    names, outfit_split, dates = set(), {}, {key: [] for key in ("train", "val", "test")}
    dimension = catalog["vectors"].shape[1]
    for row in samples:
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in names:
            raise ValueError("Sample IDs must be nonempty unique strings")
        names.add(sample_id)
        if not isinstance(row["user_id"], str) or not row["user_id"]:
            raise ValueError(f"{sample_id}: pseudonymous user_id required")
        split = row["split"]
        if split not in dates:
            raise ValueError(f"{sample_id}: invalid split")
        at = timestamp(row["timestamp"])
        dates[split].append(at)
        outfit = row["outfit_item_ids"]
        if not isinstance(outfit, list) or not outfit or len(outfit) != len(set(outfit)):
            raise ValueError(f"{sample_id}: outfit must contain distinct items")
        for item_id in outfit:
            if item_id not in catalog["index"]:
                raise ValueError(f"{sample_id}: unknown outfit item {item_id}")
        slots = [catalog["slots"][catalog["index"][key]] for key in outfit]
        if slots not in (["top", "bottom"], ["dress"]):
            raise ValueError(f"{sample_id}: expected ordered top+bottom or dress")
        key = tuple(sorted(outfit))
        if key in outfit_split and outfit_split[key] != split:
            raise ValueError(f"{sample_id}: identical outfit appears across splits")
        outfit_split[key] = split
        previous = None
        history = row["history"]
        if not isinstance(history, list):
            raise ValueError(f"{sample_id}: history must be a list")
        for event in history:
            if event["item_id"] not in catalog["index"]:
                raise ValueError(f"{sample_id}: unknown history item")
            when = timestamp(event["timestamp"])
            if when >= at or (previous is not None and when < previous):
                raise ValueError(f"{sample_id}: history must be ordered and precede target")
            previous = when
        profile = row.get("query_embedding")
        if profile is not None:
            vector = torch.tensor(profile, dtype=torch.float32)
            if vector.shape != (dimension,) or not torch.isfinite(vector).all() or vector.norm() <= 1e-8:
                raise ValueError(f"{sample_id}: invalid query embedding")
        elif not history:
            raise ValueError(f"{sample_id}: empty history needs an onboarding query embedding")
    if require_all_splits and any(not values for values in dates.values()):
        raise ValueError("Provide train, val and test samples with documented split provenance")
    populated = [dates[name] for name in ("train", "val", "test") if dates[name]]
    if any(max(left) >= min(right) for left, right in zip(populated, populated[1:])):
        raise ValueError("Splits must be strictly chronological: train < val < test")
    return {name: len(values) for name, values in dates.items()}


def load_samples(path, catalog):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    validate_samples(rows, catalog)
    return rows


def make_batch(rows, catalog, config):
    vectors, index = catalog["vectors"], catalog["index"]
    history_length = max(len(row["history"][-config.max_history:]) for row in rows)
    outfit_length = max(len(row["outfit_item_ids"]) for row in rows)
    if outfit_length > config.max_outfit:
        raise ValueError("Outfit exceeds model limits")
    b, e = len(rows), vectors.shape[1]
    history = vectors.new_zeros((b, history_length, e))
    padding = torch.ones((b, history_length), dtype=torch.bool, device=vectors.device)
    prefix = vectors.new_zeros((b, outfit_length, e))
    profiles = vectors.new_zeros((b, e))
    targets = torch.full((b, outfit_length + 1), -100, dtype=torch.long, device=vectors.device)
    for i, row in enumerate(rows):
        event_ids = [index[event["item_id"]] for event in row["history"][-config.max_history:]]
        if event_ids:
            history[i, :len(event_ids)] = vectors[event_ids]
            padding[i, :len(event_ids)] = False
        profile = row.get("query_embedding")
        if profile is not None:
            profiles[i] = torch.tensor(profile, device=vectors.device, dtype=vectors.dtype)
        elif event_ids:
            profiles[i] = vectors[event_ids].mean(0)
        else:
            raise ValueError("Empty history requires an onboarding embedding")
        if profiles[i].norm() <= 1e-8:
            raise ValueError("Profile vector cancels to zero; provide an onboarding embedding")
        outfit_ids = [index[key] for key in row["outfit_item_ids"]]
        prefix[i, :len(outfit_ids)] = vectors[outfit_ids]
        targets[i, :len(outfit_ids)] = torch.tensor(outfit_ids, device=vectors.device)
        targets[i, len(outfit_ids)] = len(vectors)  # END, not a catalog item
    return history, F.normalize(profiles, dim=-1), prefix, padding, targets
