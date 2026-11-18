"""Identical candidate pools for both next-item scorers; no aesthetic claims."""
import math

import torch

from .baseline import rank_dense
from .data import make_batch


def ranking_metrics(ranks, k=10):
    if not ranks or k < 1 or any(rank < 1 for rank in ranks):
        raise ValueError("Provide positive one-based ranks and k")
    n = len(ranks)
    return {f"hit@{k}": sum(rank <= k for rank in ranks) / n,
            f"ndcg@{k}": sum(1 / math.log2(rank + 1) if rank <= k else 0 for rank in ranks) / n,
            "mrr": sum(1 / rank for rank in ranks) / n, "predictions": n}


@torch.no_grad()
def compare_next_items(model, samples, catalog, k=10):
    """Teacher-forced next-item retrieval, using the true earlier outfit items.

    Not full FITB (no right context), free-running outfit quality, online learning
    or evidence of feedback improvement. Baseline uses the same initial profile.
    """
    if model.training:
        raise ValueError("Model must be in eval mode")
    ranks = {"clip_profile_dense": [], "clip_pog": []}
    for sample in samples:
        history, profile, prefix, padding, _ = make_batch([sample], catalog, model.config)
        scores = model.logits(history, profile, prefix, catalog["vectors"], padding)[0]
        previous = set()
        for step, target in enumerate(sample["outfit_item_ids"]):
            slot = catalog["slots"][catalog["index"][target]]
            allowed = [key for key, value in zip(catalog["ids"], catalog["slots"])
                       if value == slot and key not in previous]
            baseline = rank_dense(profile[0], catalog["vectors"], catalog["ids"],
                                  allowed_ids=allowed, limit=len(allowed))
            order = sorted(allowed, key=lambda key: (
                -float(scores[step, catalog["index"][key]]), key))
            ranks["clip_profile_dense"].append([key for key, _ in baseline].index(target) + 1)
            ranks["clip_pog"].append(order.index(target) + 1)
            previous.add(target)
    return {name: ranking_metrics(values, k) for name, values in ranks.items()}
