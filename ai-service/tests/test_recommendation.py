import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from app.recommendation.baseline import rank_dense, recommend_dense, slot_for_category, update_preference
from app.recommendation.clip_pog import ClipPOG, ModelConfig
from app.recommendation.data import make_batch, validate_samples
from app.recommendation.evaluation import compare_next_items, ranking_metrics


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = ModelConfig(embedding_dim=8, hidden_dim=16, heads=2, layers=1, dropout=0)
        self.model = ClipPOG(self.config).eval()
        self.vectors = F.normalize(torch.randn(8, 8), dim=-1)
        ids = [str(i) for i in range(8)]
        self.catalog = {"vectors": self.vectors, "ids": ids,
                        "index": {key: i for i, key in enumerate(ids)},
                        "slots": ["top", "bottom", "top", "bottom", "dress", "dress", "top", "bottom"]}
        self.rows = [{"sample_id": str(i), "user_id": "anonymous-1", "split": split,
                      "timestamp": f"2026-01-0{i + 2}T12:00:00Z",
                      "history": [{"item_id": "0", "timestamp": "2026-01-01T12:00:00Z"}],
                      "outfit_item_ids": outfit}
                     for i, (split, outfit) in enumerate([
                         ("train", ["0", "1"]), ("val", ["2", "3"]), ("test", ["4"])])]

    def test_rocchio_moves_toward_and_away(self):
        query = torch.tensor([1., 1.]); liked = torch.tensor([[1., 0.]])
        original = F.normalize(query, dim=0)
        self.assertGreater(float(update_preference(query, liked=liked)[0]), float(original[0]))
        self.assertLess(float(update_preference(query, disliked=liked)[0]), float(original[0]))

    def test_zero_feedback_result_has_safe_fallback(self):
        q = torch.tensor([1., 0.])
        from app.recommendation.baseline import RocchioWeights
        self.assertTrue(torch.equal(update_preference(q, disliked=q[None],
                            weights=RocchioWeights(disliked=1)), q))

    def test_dense_filter_and_exclusion(self):
        ranked = rank_dense(self.vectors[0], self.vectors, self.catalog["ids"],
                            allowed_ids={"0", "2"}, excluded_ids={"0"})
        self.assertEqual([item for item, _ in ranked], ["2"])
        self.assertIsNone(slot_for_category("Coat"))
        self.assertEqual(slot_for_category("Dress"), "dress")

    def test_baseline_outfit_keeps_slots(self):
        result = recommend_dense(self.vectors[0], self.catalog, excluded_ids={"0"})
        self.assertEqual([row["slot"] for row in result], ["top", "bottom"])
        self.assertNotIn("0", [row["item_id"] for row in result])

    def test_reject_zero_and_nan_queries(self):
        for q in (torch.zeros(8), torch.full((8,), float("nan"))):
            with self.assertRaises(ValueError):
                update_preference(q)

    def test_future_history_and_split_leak_are_rejected(self):
        self.assertEqual(validate_samples(self.rows, self.catalog), {"train": 1, "val": 1, "test": 1})
        rows = copy.deepcopy(self.rows)
        rows[0]["history"][0]["timestamp"] = rows[0]["timestamp"]
        with self.assertRaises(ValueError):
            validate_samples(rows, self.catalog)
        rows = copy.deepcopy(self.rows)
        rows[1]["outfit_item_ids"] = rows[0]["outfit_item_ids"]
        with self.assertRaises(ValueError):
            validate_samples(rows, self.catalog)

    def test_unknown_items_and_bad_layout_rejected(self):
        for outfit in (["missing"], ["1", "0"]):
            rows = copy.deepcopy(self.rows); rows[0]["outfit_item_ids"] = outfit
            with self.assertRaises(ValueError):
                validate_samples(rows, self.catalog)

    def test_teacher_forcing_does_not_see_future_targets(self):
        history, q, prefix, padding, _ = make_batch(self.rows[:1], self.catalog, self.config)
        original = self.model(history, q, prefix, padding)
        changed = prefix.clone(); changed[:, 1] = self.vectors[7]
        updated = self.model(history, q, changed, padding)
        self.assertTrue(torch.allclose(original[:, :2], updated[:, :2], atol=1e-6))

    def test_masked_history_does_not_change_prediction(self):
        h = self.vectors[:2][None]; q = self.vectors[:1]; prefix = self.vectors[2:3][None]
        mask = torch.tensor([[False, True]])
        first = self.model(h, q, prefix, mask)
        changed = h.clone(); changed[:, 1] = self.vectors[7] * 100
        self.assertTrue(torch.allclose(first, self.model(changed, q, prefix, mask), atol=1e-6))

    def test_empty_history_needs_real_profile(self):
        rows = copy.deepcopy(self.rows); rows[0]["history"] = []
        with self.assertRaises(ValueError):
            validate_samples(rows, self.catalog)
        rows[0]["query_embedding"] = self.vectors[6].tolist()
        validate_samples(rows, self.catalog)
        h, q, prefix, mask, _ = make_batch(rows[:1], self.catalog, self.config)
        self.assertTrue(torch.isfinite(self.model(h, q, prefix, mask)).all())

    def test_padding_targets_and_backward(self):
        h, q, prefix, padding, target = make_batch(self.rows[1:], self.catalog, self.config)
        self.assertEqual(target.tolist(), [[2, 3, 8], [4, 8, -100]])
        self.model.train()
        logits = self.model.logits(h, q, prefix, self.vectors, padding)
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=-100)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_generation_obeys_layout_exclusions_and_mode(self):
        h, q = self.vectors[:1][None], self.vectors[:1]
        result = self.model.generate(h, q, self.vectors, self.catalog["ids"],
                                    self.catalog["slots"], excluded_ids={"0"})
        self.assertEqual([row["slot"] for row in result], ["top", "bottom"])
        self.assertNotIn("0", [row["item_id"] for row in result])
        with self.assertRaises(ValueError):
            self.model.generate(h, q, self.vectors, self.catalog["ids"],
                                self.catalog["slots"], excluded_ids={"0", "2", "6"})
        self.model.train()
        with self.assertRaises(RuntimeError):
            self.model.generate(h, q, self.vectors, self.catalog["ids"], self.catalog["slots"])

    def test_metrics_and_comparison(self):
        result = ranking_metrics([1, 2, 4], k=2)
        self.assertAlmostEqual(result["hit@2"], 2/3)
        report = compare_next_items(self.model, self.rows[1:], self.catalog, k=2)
        self.assertEqual(report["clip_profile_dense"]["predictions"], 3)
        self.assertEqual(report["clip_pog"]["predictions"], 3)

    def test_training_checkpoint_and_evaluation_pipeline_with_fixture(self):
        from scripts import train_clip_pog

        # A synthetic fixture checks serialization and split flow, not quality.
        catalog = {**self.catalog, "sha256": "synthetic-catalog-for-test-only"}
        with tempfile.TemporaryDirectory(prefix="fashion-model-test-") as directory:
            root = Path(directory)
            samples = root / "samples.jsonl"
            samples.write_text("\n".join(json.dumps(row) for row in self.rows), encoding="utf-8")
            output = root / "test-checkpoint.pt"
            with patch.object(train_clip_pog, "load_catalog", return_value=catalog):
                with contextlib.redirect_stdout(io.StringIO()):
                    with patch("sys.argv", ["train_clip_pog", "--samples", str(samples),
                                           "--output", str(output), "--epochs", "1"]):
                        train_clip_pog.main()
                    checkpoint = torch.load(output, weights_only=True)
                    self.assertEqual(checkpoint["trained_steps"], 1)
                    self.assertEqual(checkpoint["catalog_sha256"], catalog["sha256"])
                    with patch("sys.argv", ["train_clip_pog", "--samples", str(samples),
                                           "--checkpoint", str(output), "--split", "test"]):
                        train_clip_pog.main()
                    samples.write_text(samples.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                    with patch("sys.argv", ["train_clip_pog", "--samples", str(samples),
                                           "--checkpoint", str(output)]):
                        with self.assertRaises(SystemExit):
                            with contextlib.redirect_stderr(io.StringIO()):
                                train_clip_pog.main()


if __name__ == "__main__":
    unittest.main()
