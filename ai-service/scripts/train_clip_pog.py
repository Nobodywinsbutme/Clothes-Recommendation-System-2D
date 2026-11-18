"""Prepare/train/evaluate the experimental CLIP+POG branch with explicit inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F

from app.recommendation.clip_pog import ClipPOG, ModelConfig
from app.recommendation.data import load_catalog, load_samples, make_batch
from app.recommendation.evaluation import compare_next_items


def loss_for(model, rows, catalog):
    history, profile, prefix, padding, targets = make_batch(rows, catalog, model.config)
    logits = model.logits(history, profile, prefix, catalog["vectors"], padding)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)


def smoke_test():
    """Check gradients/inference using generated tensors; never save weights."""
    torch.manual_seed(42)
    config = ModelConfig(embedding_dim=8, hidden_dim=16, heads=2, layers=1, dropout=0)
    model = ClipPOG(config)
    candidates = F.normalize(torch.randn(4, 8), dim=-1)
    history, profile = candidates[:1][None], candidates[0:1]
    prefix = candidates[1:3][None]
    targets = torch.tensor([[1, 2, 4]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for _ in range(3):
        optimizer.zero_grad()
        logits = model.logits(history, profile, prefix, candidates)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite synthetic loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Non-finite gradient")
        optimizer.step()
    model.eval()
    outfit = model.generate(history, profile, candidates, ["a", "b", "c", "d"],
                            ["top", "bottom", "top", "dress"])
    assert [item["slot"] for item in outfit] == ["top", "bottom"]
    print("SMOKE TEST PASSED: forward/backward and constrained generation work.")
    print("Synthetic tensors only. No quality benchmark, no saved or production model.")


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--seed-dir", type=Path, default=root / "seed")
    ap.add_argument("--samples", type=Path)
    ap.add_argument("--output", type=Path, help="New checkpoint file; existing files are not overwritten")
    ap.add_argument("--checkpoint", type=Path, help="Evaluate an existing trained checkpoint")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    if args.samples is None:
        ap.error("Provide --samples with real, documented outfit/history data; see docs/model-experiments.md")
    if min(args.epochs, args.batch_size) < 1 or not 0 < args.learning_rate < 1:
        ap.error("Positive epochs/batch-size and learning-rate in (0, 1) are required")
    if not args.validate_only and not args.checkpoint and not args.output:
        ap.error("Training requires a new --output path")
    if args.output and args.output.exists():
        ap.error("Output already exists; choose a new path")
    catalog = load_catalog(args.seed_dir)
    rows = load_samples(args.samples, catalog)
    print(json.dumps({"samples": {key: sum(r["split"] == key for r in rows)
                      for key in ("train", "val", "test")}, "catalog_sha256": catalog["sha256"]}))
    if args.validate_only:
        print("Input contract passed; this does not prove label correctness or image availability.")
        return
    catalog["vectors"] = catalog["vectors"].to(args.device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        if checkpoint.get("run_kind") != "supplied-data" or checkpoint.get("trained_steps", 0) < 1:
            ap.error("Expected a checkpoint trained on supplied data")
        if checkpoint["catalog_sha256"] != catalog["sha256"]:
            ap.error("Checkpoint and catalog differ; use the original catalog/embedding model")
        sample_hash = hashlib.sha256(args.samples.read_bytes()).hexdigest()
        if checkpoint["samples_sha256"] != sample_hash:
            ap.error("Evaluation requires the original sample file/split manifest")
        model = ClipPOG(ModelConfig(**checkpoint["config"])).to(args.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        report = compare_next_items(model, [r for r in rows if r["split"] == args.split], catalog)
        print(json.dumps({"split": args.split, "task": "teacher-forced-next-item",
                          "metrics": report}, indent=2))
        return
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = ClipPOG(ModelConfig(embedding_dim=catalog["vectors"].shape[1])).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train, val = ([r for r in rows if r["split"] == key] for key in ("train", "val"))
    best, steps = float("inf"), 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        rng.shuffle(train)
        total = 0.0
        for start in range(0, len(train), args.batch_size):
            batch = train[start:start + args.batch_size]
            optimizer.zero_grad()
            loss = loss_for(model, batch, catalog)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            steps += 1
        model.eval()
        with torch.no_grad():
            # Per-sample mean so loss is independent of batch padding/grouping.
            val_loss = sum(float(loss_for(model, [row], catalog)) for row in val) / len(val)
        print(json.dumps({"epoch": epoch + 1, "train_loss": total / len(train), "val_loss": val_loss}))
        if val_loss < best:
            best = val_loss
            torch.save({**model.metadata(), "state_dict": model.state_dict(),
                        "run_kind": "supplied-data", "trained_steps": steps,
                        "catalog_sha256": catalog["sha256"],
                        "samples_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
                        "seed": args.seed, "best_val_loss": best, "epoch": epoch + 1}, args.output)
    print(f"Saved best validation checkpoint: {args.output}")
    print("No test-set evaluation during training. Review provenance, then evaluate explicitly.")


if __name__ == "__main__":
    main()
