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
        train, val = _split_dataset(lines, train_size=80, split_seed=0)
        assert len(train) == 80
        assert len(val) == 20

    def test_no_overlap_and_full_coverage(self):
        lines = [f"item_{i}" for i in range(100)]
        train, val = _split_dataset(lines, train_size=80, split_seed=0)
        assert set(train).isdisjoint(val)
        assert set(train) | set(val) == set(lines)

    def test_deterministic_same_seed(self):
        lines = [f"item_{i}" for i in range(100)]
        train_a, val_a = _split_dataset(lines, train_size=80, split_seed=7)
        train_b, val_b = _split_dataset(lines, train_size=80, split_seed=7)
        assert train_a == train_b
        assert val_a == val_b

    def test_different_seed_gives_different_split(self):
        lines = [f"item_{i}" for i in range(100)]
        train_a, _ = _split_dataset(lines, train_size=80, split_seed=1)
        train_b, _ = _split_dataset(lines, train_size=80, split_seed=2)
        assert train_a != train_b

    def test_input_list_not_mutated(self):
        lines = [f"item_{i}" for i in range(100)]
        original = list(lines)
        _split_dataset(lines, train_size=80, split_seed=0)
        assert lines == original

    def test_actually_shuffles(self):
        # With a shuffle, the train split should not simply be the first
        # train_size elements of the input.
        lines = [f"item_{i}" for i in range(100)]
        train, _ = _split_dataset(lines, train_size=80, split_seed=0)
        assert train != lines[:80]

    def test_minimal_val_split(self):
        lines = [f"item_{i}" for i in range(10)]
        train, val = _split_dataset(lines, train_size=9, split_seed=0)
        assert len(train) == 9
        assert len(val) == 1

    @pytest.mark.parametrize(
        "train_size",
        [0, -1, 100, 101, 200],
        ids=["zero", "negative", "equal_to_len", "len_plus_one", "too_large"],
    )
    def test_invalid_train_size_raises(self, train_size):
        lines = [f"item_{i}" for i in range(100)]
        with pytest.raises(ValueError, match="train_size"):
            _split_dataset(lines, train_size=train_size, split_seed=0)

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="train_size"):
            _split_dataset([], train_size=1, split_seed=0)
