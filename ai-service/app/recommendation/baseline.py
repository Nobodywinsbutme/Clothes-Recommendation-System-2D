"""CLIP/Rocchio reference scorer, plus a Qdrant candidate-search adapter.

The dense scorer is useful for controlled experiments without a running DB.
Neither cosine scores nor these slot rules are learned outfit compatibility.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class RocchioWeights:
    original: float = 1.0
    liked: float = 0.8
    disliked: float = 0.4


def update_preference(original, liked=None, disliked=None, weights=RocchioWeights()):
    """Recompute from the original query and all scoped feedback, not recursively.

    The caller resolves outfit feedback/reasons first. An outfit dislike must
    never be expanded automatically into dislikes of each individual item.
    """
    if not all(math.isfinite(x) and x >= 0 for x in vars(weights).values()):
        raise ValueError("Rocchio weights must be finite and nonnegative")
    if original.ndim != 1 or not torch.isfinite(original).all() or original.norm() <= 1e-8:
        raise ValueError("Expected a finite nonzero query vector")
    original = F.normalize(original.float(), dim=0)
    result = weights.original * original
    for vectors, weight in ((liked, weights.liked), (disliked, -weights.disliked)):
        if vectors is None or vectors.numel() == 0:
            continue
        if vectors.ndim != 2 or vectors.shape[1] != len(original):
            raise ValueError("Feedback vector dimension does not match query")
        if not torch.isfinite(vectors).all() or (vectors.norm(dim=-1) <= 1e-8).any():
            raise ValueError("Feedback vectors must be finite and nonzero")
        result = result + weight * F.normalize(vectors.to(original), dim=-1).mean(0)
    return original if result.norm() <= 1e-8 else F.normalize(result, dim=0)


def slot_for_category(category):
    """Explicit MVP taxonomy. Coats/robes/kimono are not dress substitutes."""
    if category in {"Dress", "Sundress", "Shirtdress"}:
        return "dress"
    if category in {"Blouse", "Button-Down", "Tank", "Top", "Tee", "Sweater",
                    "Flannel", "Hoodie", "Henley", "Jersey", "Turtleneck", "Halter"}:
        return "top"
    if category in {"Skirt", "Jeans", "Shorts", "Joggers", "Sweatpants", "Jeggings",
                    "Cutoffs", "Sweatshorts", "Leggings", "Culottes", "Chinos",
                    "Gauchos", "Jodhpurs", "Capris"}:
        return "bottom"
    return None


def rank_dense(query, vectors, item_ids, *, allowed_ids=None, excluded_ids=(), limit=10):
    if limit < 1:
        raise ValueError("limit must be positive")
    if vectors.ndim != 2 or vectors.shape != (len(item_ids), query.numel()):
        raise ValueError("Catalog and query shapes do not match")
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Catalog IDs must be unique")
    query = update_preference(query)
    if not torch.isfinite(vectors).all() or (vectors.norm(dim=-1) <= 1e-8).any():
        raise ValueError("Catalog vectors must be finite and nonzero")
    scores = F.normalize(vectors.to(query), dim=-1) @ query
    excluded = set(excluded_ids)
    allowed = None if allowed_ids is None else set(allowed_ids)
    candidates = [(item_id, float(scores[i])) for i, item_id in enumerate(item_ids)
                  if item_id not in excluded and (allowed is None or item_id in allowed)]
    return sorted(candidates, key=lambda row: (-row[1], str(row[0])))[:limit]


def search_qdrant(client, collection, query, *, category, excluded_ids=(), limit=10):
    """Keyword-only explicit category; does not hide Qdrant errors with fallback."""
    from qdrant_client import models

    if limit < 1 or not category:
        raise ValueError("A category and positive limit are required")
    query = update_preference(query)
    filters = models.Filter(
        must=[models.FieldCondition(key="category", match=models.MatchValue(value=category))],
        must_not=[models.HasIdCondition(has_id=[int(key) if isinstance(key, str) and key.isdecimal()
                                               else key for key in excluded_ids])] if excluded_ids else None,
    )
    return client.query_points(collection_name=collection, query=query.tolist(),
                               query_filter=filters, limit=limit, with_payload=True).points


def recommend_dense(query, catalog, *, layout=("top", "bottom"), excluded_ids=()):
    """Reference baseline outfit with fixed slot rules, without learned compatibility."""
    if tuple(layout) not in (("top", "bottom"), ("dress",)):
        raise ValueError("MVP layouts are top+bottom or dress")
    excluded, outfit = set(excluded_ids), []
    for slot in layout:
        allowed = [key for key, value in zip(catalog["ids"], catalog["slots"]) if value == slot]
        ranked = rank_dense(query, catalog["vectors"], catalog["ids"],
                            allowed_ids=allowed, excluded_ids=excluded, limit=1)
        if not ranked:
            raise ValueError(f"No eligible candidates for slot {slot}")
        key, score = ranked[0]
        outfit.append({"item_id": key, "slot": slot, "score": score})
        excluded.add(key)
    return outfit
