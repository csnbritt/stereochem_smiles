"""Tests for _split_dataset in scripts/generate_manuscript_data.py."""

import sys
from pathlib import Path

import pytest

pytest.importorskip(
    "transformers",
    reason="generate_manuscript_data imports the trainer stack",
)

_SCRIPTS_DIR = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from generate_manuscript_data import _split_dataset


class TestSplitDataset:
    """Tests for the seeded train/val split helper."""

    def test_split_sizes(self):
        lines = [f"item_{i}" for i in range(100)]
        train, val = _split_dataset(lines, train_frac=0.8, split_seed=0)
        assert len(train) == 80
        assert len(val) == 20

    def test_no_overlap_and_full_coverage(self):
        lines = [f"item_{i}" for i in range(100)]
        train, val = _split_dataset(lines, train_frac=0.8, split_seed=0)
        assert set(train).isdisjoint(val)
        assert set(train) | set(val) == set(lines)

    def test_deterministic_same_seed(self):
        lines = [f"item_{i}" for i in range(100)]
        train_a, val_a = _split_dataset(lines, train_frac=0.8, split_seed=7)
        train_b, val_b = _split_dataset(lines, train_frac=0.8, split_seed=7)
        assert train_a == train_b
        assert val_a == val_b

    def test_different_seed_gives_different_split(self):
        lines = [f"item_{i}" for i in range(100)]
        train_a, _ = _split_dataset(lines, train_frac=0.8, split_seed=1)
        train_b, _ = _split_dataset(lines, train_frac=0.8, split_seed=2)
        assert train_a != train_b

    def test_input_list_not_mutated(self):
        lines = [f"item_{i}" for i in range(100)]
        original = list(lines)
        _split_dataset(lines, train_frac=0.8, split_seed=0)
        assert lines == original

    def test_actually_shuffles(self):
        # With a shuffle, the train split should not simply be the first
        # train_size elements of the input.
        lines = [f"item_{i}" for i in range(100)]
        train, _ = _split_dataset(lines, train_frac=0.8, split_seed=0)
        assert train != lines[:80]

    def test_minimal_val_split(self):
        lines = [f"item_{i}" for i in range(10)]
        train, val = _split_dataset(lines, train_frac=0.9, split_seed=0)
        assert len(train) == 9
        assert len(val) == 1

    def test_frac_rounds_down(self):
        # int(7 * 0.9) = 6, so val gets the remainder
        lines = [f"item_{i}" for i in range(7)]
        train, val = _split_dataset(lines, train_frac=0.9, split_seed=0)
        assert len(train) == 6
        assert len(val) == 1

    @pytest.mark.parametrize(
        "train_frac",
        [0.0, -0.1, 1.0, 1.5],
        ids=["zero", "negative", "one", "over_one"],
    )
    def test_out_of_range_frac_raises(self, train_frac):
        lines = [f"item_{i}" for i in range(100)]
        with pytest.raises(ValueError, match="train_frac"):
            _split_dataset(lines, train_frac=train_frac, split_seed=0)

    def test_tiny_frac_empty_train_raises(self):
        # int(100 * 0.005) = 0 -> empty train split
        lines = [f"item_{i}" for i in range(100)]
        with pytest.raises(ValueError, match="train_frac"):
            _split_dataset(lines, train_frac=0.005, split_seed=0)

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="train_frac"):
            _split_dataset([], train_frac=0.9, split_seed=0)
