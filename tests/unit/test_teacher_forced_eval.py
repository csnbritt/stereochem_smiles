"""Tests for teacher-forced eval helpers in models/training/trainer.py."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="trainer imports torch")
pytest.importorskip(
    "transformers",
    reason="trainer imports the transformers stack",
)

from models.tokenizer.tokenizer import CustomTokenizer
from models.tokenizer.vocab import smiles_token_to_id_dict
from models.training.trainer import (
    SmilesTrainer,
    _argmax_logits_for_metrics,
    _truncate_at_eos,
)

VOCAB = smiles_token_to_id_dict
EOS_ID = VOCAB["$"]
PAD_ID = VOCAB["[PAD]"]


def _ids(tokens: list[str]) -> list[int]:
    return [VOCAB[t] for t in tokens]


class TestArgmaxLogitsForMetrics:
    """Tests for the preprocess_logits_for_metrics hook."""

    def test_returns_argmax_ids(self):
        logits = torch.zeros(2, 3, 10)
        logits[0, 0, 3] = 1.0
        logits[0, 1, 7] = 1.0
        logits[0, 2, 0] = 1.0
        logits[1, 0, 5] = 1.0
        logits[1, 1, 5] = 1.0
        logits[1, 2, 9] = 1.0

        result = _argmax_logits_for_metrics(logits, labels=None)

        assert result.tolist() == [[3, 7, 0], [5, 5, 9]]

    def test_unwraps_tuple_logits(self):
        logits = torch.zeros(1, 2, 5)
        logits[0, 0, 2] = 1.0
        logits[0, 1, 4] = 1.0

        result = _argmax_logits_for_metrics((logits, "extra"), labels=None)

        assert result.tolist() == [[2, 4]]


class TestTruncateAtEos:
    """Tests for EOS truncation of argmax predictions."""

    def test_truncates_at_first_eos(self):
        ids = np.array(_ids(["C", "C", "O", "$", "C", "C"]))
        result = _truncate_at_eos(ids, EOS_ID)
        assert result.tolist() == _ids(["C", "C", "O", "$"])

    def test_no_eos_returns_unchanged(self):
        ids = np.array(_ids(["C", "C", "O"]))
        result = _truncate_at_eos(ids, EOS_ID)
        assert result.tolist() == ids.tolist()

    def test_eos_at_position_zero(self):
        ids = np.array(_ids(["$", "C", "O"]))
        result = _truncate_at_eos(ids, EOS_ID)
        assert result.tolist() == [EOS_ID]

    def test_multiple_eos_uses_first(self):
        ids = np.array(_ids(["C", "$", "O", "$", "C"]))
        result = _truncate_at_eos(ids, EOS_ID)
        assert result.tolist() == _ids(["C", "$"])


class TestComputeMetricsTeacherForced:
    """compute_metrics on argmax-style predictions (greedy-equivalent)."""

    def _make_trainer(self) -> SmilesTrainer:
        trainer = SmilesTrainer.__new__(SmilesTrainer)
        trainer.tokenizer = CustomTokenizer(smiles_token_to_id_dict)
        trainer.trainer = None
        trainer.analysis_dir = ""
        return trainer

    def test_exact_match_ignores_junk_after_eos(self):
        # Argmax emits ids past the predicted EOS; greedy decoding would
        # have stopped, so this must still count as an exact match.
        trainer = self._make_trainer()
        pred = np.array([_ids(["C", "C", "O", "$", "C", "C"])])
        label = np.array([_ids(["C", "C", "O", "$"]) + [-100, -100]])

        metrics = trainer.compute_metrics((pred, label))

        assert metrics["exact_match"] == 1.0
        assert metrics["token_accuracy"] == 1.0

    def test_wrong_token_not_exact_match(self):
        trainer = self._make_trainer()
        pred = np.array([_ids(["C", "C", "C", "$"])])
        label = np.array([_ids(["C", "C", "O", "$"])])

        metrics = trainer.compute_metrics((pred, label))

        assert metrics["exact_match"] == 0.0
        # 3 of 4 label tokens correct (C, C, $)
        assert metrics["token_accuracy"] == pytest.approx(0.75)

    def test_missing_eos_not_exact_match(self):
        # No EOS argmaxed -> greedy would overrun -> not an exact match.
        trainer = self._make_trainer()
        pred = np.array([_ids(["C", "C", "O", "C"])])
        label = np.array([_ids(["C", "C", "O", "$"])])

        metrics = trainer.compute_metrics((pred, label))

        assert metrics["exact_match"] == 0.0

    def test_early_eos_not_exact_match(self):
        # EOS argmaxed early -> greedy stops early -> wrong sequence.
        trainer = self._make_trainer()
        pred = np.array([_ids(["C", "$", "O", "$"])])
        label = np.array([_ids(["C", "C", "O", "$"])])

        metrics = trainer.compute_metrics((pred, label))

        assert metrics["exact_match"] == 0.0
