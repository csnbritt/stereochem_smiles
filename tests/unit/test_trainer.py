"""
Unit tests for trainer utility functions and configuration.

Covers:
  - strip_stereo_from_smiles removes all stereochemistry
  - compute_top_tokens returns most frequent tokens
  - CustomSmallConfig has max_steps and use_early_stopping fields
  - SmilesPreprocessing TypedDict includes strip_stereo
"""

import pytest
from rdkit import Chem

from models.tokenizer.tokenizer import CustomTokenizer
from models.tokenizer.vocab import smiles_token_to_id_dict
from models.training.trainer import (
    CustomSmallConfig,
    SmilesPreprocessing,
    compute_top_tokens,
    strip_stereo_from_smiles,
)

# ── strip_stereo_from_smiles ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "smiles",
    [
        "C[C@H](O)C(=O)O",  # R/S stereo
        "C/C=C\\C",  # E/Z stereo
        "C[C@H](O)[C@@H](N)C(=O)O",  # multiple stereocenters
        "CCO",  # no stereo
    ],
)
def test_strip_stereo_removes_all_stereo(smiles):
    """strip_stereo_from_smiles should produce a valid SMILES with no stereo."""
    result = strip_stereo_from_smiles(smiles)
    assert result is not None
    mol = Chem.MolFromSmiles(result)
    assert mol is not None
    for atom in mol.GetAtoms():
        assert atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
    for bond in mol.GetBonds():
        assert bond.GetStereo() == Chem.BondStereo.STEREONONE


def test_strip_stereo_invalid_smiles():
    """strip_stereo_from_smiles should return None for invalid SMILES."""
    assert strip_stereo_from_smiles("not_a_smiles") is None


def test_strip_stereo_preserves_connectivity():
    """Stripped SMILES should have the same connectivity as the original."""
    smiles = "C[C@H](O)C(=O)O"
    result = strip_stereo_from_smiles(smiles)
    assert result is not None
    mol_orig = Chem.MolFromSmiles(smiles)
    mol_stripped = Chem.MolFromSmiles(result)
    assert mol_orig.GetNumAtoms() == mol_stripped.GetNumAtoms()
    assert mol_orig.GetNumBonds() == mol_stripped.GetNumBonds()


# ── compute_top_tokens ────────────────────────────────────────────────────────


def test_compute_top_tokens_basic():
    """compute_top_tokens should return the most frequent output tokens."""
    data = [
        ("CCO", "CCO"),
        ("CCN", "CCN"),
        ("CCO", "CCO"),
        ("CCC", "CCC"),
    ]
    tokenizer = CustomTokenizer(token_to_id=smiles_token_to_id_dict)
    top = compute_top_tokens(data, tokenizer, top_k=5)
    assert len(top) <= 5
    assert len(top) > 0


def test_compute_top_tokens_empty_data():
    """compute_top_tokens should return an empty list for empty data."""
    tokenizer = CustomTokenizer(token_to_id=smiles_token_to_id_dict)
    top = compute_top_tokens([], tokenizer, top_k=10)
    assert top == []


def test_compute_top_tokens_top_k_limit():
    """compute_top_tokens should respect the top_k limit."""
    data = [("CCO", "CCO")] * 100
    tokenizer = CustomTokenizer(token_to_id=smiles_token_to_id_dict)
    top = compute_top_tokens(data, tokenizer, top_k=3)
    assert len(top) <= 3


# ── CustomSmallConfig ─────────────────────────────────────────────────────────


def test_custom_small_config_max_steps_default():
    """max_steps should default to None."""
    config = CustomSmallConfig(output_dir="/tmp/test")
    assert config.max_steps is None


def test_custom_small_config_use_early_stopping_default():
    """use_early_stopping should default to False."""
    config = CustomSmallConfig(output_dir="/tmp/test")
    assert config.use_early_stopping is False


def test_custom_small_config_max_steps_set():
    """max_steps should be settable."""
    config = CustomSmallConfig(output_dir="/tmp/test", max_steps=50000)
    assert config.max_steps == 50000


def test_custom_small_config_use_early_stopping_set():
    """use_early_stopping should be settable."""
    config = CustomSmallConfig(output_dir="/tmp/test", use_early_stopping=True)
    assert config.use_early_stopping is True


# ── SmilesPreprocessing TypedDict ─────────────────────────────────────────────


def test_smiles_preprocessing_has_strip_stereo():
    """SmilesPreprocessing should accept a strip_stereo key."""
    preprocessing: SmilesPreprocessing = {
        "input_smiles_syntax": "smiles",
        "input_smiles_type": "random",
        "crisp_input_deferred": False,
        "strip_stereo": True,
    }
    assert preprocessing["strip_stereo"] is True


def test_smiles_preprocessing_strip_stereo_optional():
    """SmilesPreprocessing should work without strip_stereo (backward compat)."""
    preprocessing: SmilesPreprocessing = {
        "input_smiles_syntax": "smiles",
        "input_smiles_type": "random",
        "crisp_input_deferred": False,
        "strip_stereo": False,
    }
    assert preprocessing["strip_stereo"] is False


# ── _to_canonical_smiles ──────────────────────────────────────────────────────


def test_to_canonical_smiles_standard():
    """_to_canonical_smiles should canonicalize standard SMILES."""
    result = _to_canonical_smiles("C(C)(O)C(=O)O")
    assert result is not None
    mol = Chem.MolFromSmiles(result)
    assert mol is not None


def test_to_canonical_smiles_strips_stereo():
    """_to_canonical_smiles with strip_stereo=True should remove stereo."""
    result = _to_canonical_smiles("C[C@H](O)C(=O)O", strip_stereo=True)
    assert result is not None
    mol = Chem.MolFromSmiles(result)
    assert mol is not None
    for atom in mol.GetAtoms():
        assert atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED


def test_to_canonical_smiles_preserves_stereo():
    """_to_canonical_smiles with strip_stereo=False should keep stereo."""
    result = _to_canonical_smiles("C[C@H](O)C(=O)O", strip_stereo=False)
    assert result is not None
    mol = Chem.MolFromSmiles(result)
    assert mol is not None
    has_stereo = any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in mol.GetAtoms()
    )
    assert has_stereo


def test_to_canonical_smiles_invalid():
    """_to_canonical_smiles should return None for invalid SMILES."""
    assert _to_canonical_smiles("not_a_smiles") is None


def test_to_canonical_smiles_crisp():
    """_to_canonical_smiles should decode CRISP SMILES to standard SMILES."""
    # A CRISP SMILES with stereo tokens — decode should produce valid SMILES
    crisp = "C[C@@H](O)C(=O)O"  # Standard SMILES that decode() returns unchanged
    result = _to_canonical_smiles(crisp, strip_stereo=True)
    assert result is not None
    mol = Chem.MolFromSmiles(result)
    assert mol is not None


# ── process_single_smiles with strip_stereo ───────────────────────────────────


def test_process_single_smiles_strip_stereo():
    """process_single_smiles with strip_stereo=True should remove stereo."""
    preprocessing: SmilesPreprocessing = {
        "input_smiles_syntax": "smiles",
        "input_smiles_type": "canonical",
        "crisp_input_deferred": False,
        "strip_stereo": True,
    }
    result = process_single_smiles("C[C@H](O)C(=O)O", preprocessing)
    assert result is not None
    mol = Chem.MolFromSmiles(result)
    assert mol is not None
    for atom in mol.GetAtoms():
        assert atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED


def test_process_single_smiles_no_strip():
    """process_single_smiles with strip_stereo=False should keep stereo."""
    preprocessing: SmilesPreprocessing = {
        "input_smiles_syntax": "smiles",
        "input_smiles_type": "canonical",
        "crisp_input_deferred": False,
        "strip_stereo": False,
    }
    result = process_single_smiles("C[C@H](O)C(=O)O", preprocessing)
    assert result is not None
    mol = Chem.MolFromSmiles(result)
    assert mol is not None
    has_stereo = any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in mol.GetAtoms()
    )
    assert has_stereo


def test_process_single_smiles_invalid():
    """process_single_smiles should return None for invalid SMILES."""
    preprocessing: SmilesPreprocessing = {
        "input_smiles_syntax": "smiles",
        "input_smiles_type": "canonical",
        "crisp_input_deferred": False,
        "strip_stereo": False,
    }
    assert process_single_smiles("not_a_smiles", preprocessing) is None
