"""
Unit tests for internal methods of CRISPSmiles.

Tests each private method in isolation, covering:
  - Correct transformation of valid inputs
  - Edge cases (empty, no stereo, mixed stereo types)
  - Selectivity (R/S methods ignore E/Z isotopes and vice versa)
  - Invalid input handling
  - Dead-code methods (verify they still function if called directly)
"""

import pytest
from rdkit import Chem

from crisp_smiles.main import (
    STEREO_E_ISOTOPE,
    STEREO_R_ISOTOPE,
    STEREO_S_ISOTOPE,
    STEREO_Z_ISOTOPE,
    CRISPSmiles,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def converter() -> CRISPSmiles:
    return CRISPSmiles()


def _to_canonical(smiles: str) -> str:
    """Helper: canonicalize SMILES via RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


# ── _generate_local_stereochem_syntax_for_mol ─────────────────────────────────


class TestGenerateLocalStereochemSyntax:
    """Tests for the isotope-encoding step."""

    def test_rs_stereo_produces_isotope_labels(self, converter: CRISPSmiles):
        """R/S stereo should be encoded as isotope 10001 or 10002."""
        result = converter._generate_local_stereochem_syntax_for_mol(
            "C[C@H](O)C(=O)O", smiles_mode="canonical"
        )
        assert result is not None
        assert "@" not in result
        assert "\\" not in result
        assert "/" not in result
        # Should contain one of the R/S isotopes
        assert str(STEREO_R_ISOTOPE) in result or str(STEREO_S_ISOTOPE) in result

    def test_ez_stereo_produces_isotope_labels(self, converter: CRISPSmiles):
        """E/Z stereo should be encoded as isotope 10003 or 10004."""
        result = converter._generate_local_stereochem_syntax_for_mol(
            "C/C=C/C", smiles_mode="canonical"
        )
        assert result is not None
        assert "/" not in result
        assert "\\" not in result
        assert str(STEREO_E_ISOTOPE) in result or str(STEREO_Z_ISOTOPE) in result

    def test_both_rs_and_ez(self, converter: CRISPSmiles):
        """Both R/S and E/Z should be encoded simultaneously."""
        result = converter._generate_local_stereochem_syntax_for_mol(
            "C/C=C/[C@H](O)C(=O)O", smiles_mode="canonical"
        )
        assert result is not None
        has_rs = str(STEREO_R_ISOTOPE) in result or str(STEREO_S_ISOTOPE) in result
        has_ez = str(STEREO_E_ISOTOPE) in result or str(STEREO_Z_ISOTOPE) in result
        assert has_rs
        assert has_ez

    def test_no_stereo_no_isotopes(self, converter: CRISPSmiles):
        """Molecules without stereo should not have stereo isotopes."""
        result = converter._generate_local_stereochem_syntax_for_mol(
            "CCO", smiles_mode="canonical"
        )
        assert result is not None
        assert str(STEREO_R_ISOTOPE) not in result
        assert str(STEREO_S_ISOTOPE) not in result
        assert str(STEREO_E_ISOTOPE) not in result
        assert str(STEREO_Z_ISOTOPE) not in result

    def test_invalid_smiles_returns_none(self, converter: CRISPSmiles):
        result = converter._generate_local_stereochem_syntax_for_mol("invalid!!")
        assert result is None

    def test_random_mode_produces_valid_smiles(self, converter: CRISPSmiles):
        result = converter._generate_local_stereochem_syntax_for_mol(
            "C[C@H](O)C(=O)O", smiles_mode="random"
        )
        assert result is not None
        mol = Chem.MolFromSmiles(result)
        assert mol is not None

    def test_invalid_mode_falls_back_to_canonical(self, converter: CRISPSmiles):
        result = converter._generate_local_stereochem_syntax_for_mol(
            "C[C@H](O)C(=O)O", smiles_mode="bad_mode"
        )
        can = converter._generate_local_stereochem_syntax_for_mol(
            "C[C@H](O)C(=O)O", smiles_mode="canonical"
        )
        assert result is not None
        assert can is not None
        assert result == can

    def test_strips_directional_bonds(self, converter: CRISPSmiles):
        """Output should never contain / or \\ characters."""
        result = converter._generate_local_stereochem_syntax_for_mol(
            "C/C=C\\C/C=C/C", smiles_mode="canonical"
        )
        assert result is not None
        assert "/" not in result
        assert "\\" not in result


# ── _revert_all_stereochem_to_global ──────────────────────────────────────────


class TestRevertAllStereochemToGlobal:
    """Tests for the isotope-to-stereo restoration step."""

    def test_revert_rs_only(self, converter: CRISPSmiles):
        """R/S isotopes should be restored to @/@@ notation."""
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            "C[C@H](O)C(=O)O", smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._revert_all_stereochem_to_global(intermediate)
        can_orig = _to_canonical("C[C@H](O)C(=O)O")
        can_result = _to_canonical(result)
        assert can_orig == can_result

    def test_revert_ez_only(self, converter: CRISPSmiles):
        """E/Z isotopes should be restored to /\\ notation."""
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            "C/C=C/C", smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._revert_all_stereochem_to_global(intermediate)
        can_orig = _to_canonical("C/C=C/C")
        can_result = _to_canonical(result)
        assert can_orig == can_result

    def test_revert_both_rs_and_ez(self, converter: CRISPSmiles):
        """Both R/S and E/Z isotopes should be restored simultaneously."""
        smiles = "C/C=C/[C@H](O)C(=O)O"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._revert_all_stereochem_to_global(intermediate)
        can_orig = _to_canonical(smiles)
        can_result = _to_canonical(result)
        assert can_orig == can_result

    def test_revert_no_stereo_isotopes(self, converter: CRISPSmiles):
        """SMILES without stereo isotopes should round-trip through RDKit."""
        result = converter._revert_all_stereochem_to_global("CCO")
        can = _to_canonical("CCO")
        assert _to_canonical(result) == can

    def test_revert_multi_center(self, converter: CRISPSmiles):
        """Multiple R/S centers should all be restored."""
        smiles = "O[C@H](C(=O)O)[C@@H](O)C(=O)O"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._revert_all_stereochem_to_global(intermediate)
        can_orig = _to_canonical(smiles)
        can_result = _to_canonical(result)
        assert can_orig == can_result


# ── _convert_syntax_non_deferred ──────────────────────────────────────────────


class TestConvertSyntaxNonDeferred:
    """Tests for R/S isotope → inline token conversion."""

    def test_converts_rs_isotopes_to_tokens(self, converter: CRISPSmiles):
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            "C[C@H](O)C(=O)O", smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._convert_syntax_non_deferred(intermediate)
        assert "[STEREO_R]" in result or "[STEREO_S]" in result

    def test_ignores_ez_isotopes(self, converter: CRISPSmiles):
        """R/S conversion should not touch E/Z isotopes."""
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            "C/C=C/C", smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._convert_syntax_non_deferred(intermediate)
        assert "[STEREO_R]" not in result
        assert "[STEREO_S]" not in result
        # E/Z isotopes should still be present as numbers
        assert str(STEREO_E_ISOTOPE) in result or str(STEREO_Z_ISOTOPE) in result

    def test_no_isotopes_unchanged(self, converter: CRISPSmiles):
        result = converter._convert_syntax_non_deferred("CCO")
        assert result == "CCO"

    def test_multiple_rs_centers(self, converter: CRISPSmiles):
        smiles = "O[C@H](C(=O)O)[C@@H](O)C(=O)O"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._convert_syntax_non_deferred(intermediate)
        r_count = result.count("[STEREO_R]") + result.count("[STEREO_S]")
        assert r_count == 2


# ── _invert_syntax_non_deferred ────────────────────────────────────────────────


class TestInvertSyntaxNonDeferred:
    """Tests for inline R/S token → isotope label conversion."""

    def test_inverts_rs_tokens_to_isotopes(self, converter: CRISPSmiles):
        # First convert to non-deferred, then invert back
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            "C[C@H](O)C(=O)O", smiles_mode="canonical"
        )
        assert intermediate is not None
        converted = converter._convert_syntax_non_deferred(intermediate)
        inverted = converter._invert_syntax_non_deferred(converted)
        # Should contain isotope numbers
        assert str(STEREO_R_ISOTOPE) in inverted or str(STEREO_S_ISOTOPE) in inverted

    def test_no_tokens_unchanged(self, converter: CRISPSmiles):
        result = converter._invert_syntax_non_deferred("CCO")
        assert result == "CCO"

    def test_ignores_ez_tokens(self, converter: CRISPSmiles):
        """R/S inversion should not touch [STEREO_E]/[STEREO_Z] tokens."""
        smiles = "C/C=C/C"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        converted = converter._convert_db_syntax_non_deferred(intermediate)
        inverted = converter._invert_syntax_non_deferred(converted)
        assert "[STEREO_E]" in inverted or "[STEREO_Z]" in inverted

    def test_roundtrip_invert(self, converter: CRISPSmiles):
        """convert → invert should recover the original isotope-labeled SMILES."""
        smiles = "C[C@H](O)C(=O)O"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        converted = converter._convert_syntax_non_deferred(intermediate)
        inverted = converter._invert_syntax_non_deferred(converted)
        # The isotope-labeled SMILES should be equivalent
        assert _to_canonical(inverted) == _to_canonical(intermediate)


# ── _convert_syntax_deferred ──────────────────────────────────────────────────


class TestConvertSyntaxDeferred:
    """Tests for R/S isotope → deferred placeholder conversion."""

    def test_produces_pipe_separator(self, converter: CRISPSmiles):
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            "C[C@H](O)C(=O)O", smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._convert_syntax_deferred(intermediate)
        assert "|" in result

    def test_produces_stereo_center_placeholders(self, converter: CRISPSmiles):
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            "C[C@H](O)C(=O)O", smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._convert_syntax_deferred(intermediate)
        assert "[STEREO_CENTER_" in result

    def test_no_isotopes_no_pipe(self, converter: CRISPSmiles):
        result = converter._convert_syntax_deferred("CCO")
        assert "|" not in result
        assert result == "CCO"

    def test_ignores_ez_isotopes(self, converter: CRISPSmiles):
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            "C/C=C/C", smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._convert_syntax_deferred(intermediate)
        assert "[STEREO_CENTER_" not in result
        assert "|" not in result

    def test_multiple_rs_centers_numbered_sequentially(self, converter: CRISPSmiles):
        smiles = "O[C@H](C(=O)O)[C@@H](O)C(=O)O"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._convert_syntax_deferred(intermediate)
        assert "[STEREO_CENTER_1]" in result
        assert "[STEREO_CENTER_2]" in result


# ── _invert_syntax_deferred ────────────────────────────────────────────────────


class TestInvertSyntaxDeferred:
    """Tests for deferred R/S → isotope label inversion."""

    def test_inverts_deferred_rs_to_isotopes(self, converter: CRISPSmiles):
        smiles = "C[C@H](O)C(=O)O"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        converted = converter._convert_syntax_deferred(intermediate)
        inverted = converter._invert_syntax_deferred(converted)
        assert str(STEREO_R_ISOTOPE) in inverted or str(STEREO_S_ISOTOPE) in inverted

    def test_no_pipe_returns_unchanged(self, converter: CRISPSmiles):
        result = converter._invert_syntax_deferred("CCO")
        assert result == "CCO"

    def test_preserves_ez_deferred_assignments(self, converter: CRISPSmiles):
        """R/S inversion should leave [DB_STEREO_N] assignments in the | block."""
        smiles = "C/C=C/[C@H](O)C(=O)O"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        rs_deferred = converter._convert_syntax_deferred(intermediate)
        ez_deferred = converter._convert_db_syntax_deferred(rs_deferred)
        # Now invert only R/S
        inverted = converter._invert_syntax_deferred(ez_deferred)
        # E/Z deferred assignments should still be present
        assert "[DB_STEREO_" in inverted
        assert "|" in inverted

    def test_roundtrip_invert(self, converter: CRISPSmiles):
        smiles = "C[C@H](O)C(=O)O"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        converted = converter._convert_syntax_deferred(intermediate)
        inverted = converter._invert_syntax_deferred(converted)
        assert _to_canonical(inverted) == _to_canonical(intermediate)


# ── _convert_db_syntax_non_deferred ───────────────────────────────────────────


class TestConvertDbSyntaxNonDeferred:
    """Tests for E/Z isotope → inline token conversion."""

    def test_converts_ez_isotopes_to_tokens(self, converter: CRISPSmiles):
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            "C/C=C/C", smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._convert_db_syntax_non_deferred(intermediate)
        assert "[STEREO_E]" in result or "[STEREO_Z]" in result

    def test_ignores_rs_isotopes(self, converter: CRISPSmiles):
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            "C[C@H](O)C(=O)O", smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._convert_db_syntax_non_deferred(intermediate)
        assert "[STEREO_E]" not in result
        assert "[STEREO_Z]" not in result
        assert str(STEREO_R_ISOTOPE) in result or str(STEREO_S_ISOTOPE) in result

    def test_no_isotopes_unchanged(self, converter: CRISPSmiles):
        result = converter._convert_db_syntax_non_deferred("CCO")
        assert result == "CCO"


# ── _invert_db_syntax_non_deferred ─────────────────────────────────────────────


class TestInvertDbSyntaxNonDeferred:
    """Tests for inline E/Z token → isotope label conversion."""

    def test_inverts_ez_tokens_to_isotopes(self, converter: CRISPSmiles):
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            "C/C=C/C", smiles_mode="canonical"
        )
        assert intermediate is not None
        converted = converter._convert_db_syntax_non_deferred(intermediate)
        inverted = converter._invert_db_syntax_non_deferred(converted)
        assert str(STEREO_E_ISOTOPE) in inverted or str(STEREO_Z_ISOTOPE) in inverted

    def test_no_tokens_unchanged(self, converter: CRISPSmiles):
        result = converter._invert_db_syntax_non_deferred("CCO")
        assert result == "CCO"

    def test_ignores_rs_tokens(self, converter: CRISPSmiles):
        """E/Z inversion should not touch [STEREO_R]/[STEREO_S] tokens."""
        smiles = "C[C@H](O)C(=O)O"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        converted = converter._convert_syntax_non_deferred(intermediate)
        inverted = converter._invert_db_syntax_non_deferred(converted)
        assert "[STEREO_R]" in inverted or "[STEREO_S]" in inverted

    def test_roundtrip_invert(self, converter: CRISPSmiles):
        smiles = "C/C=C/C"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        converted = converter._convert_db_syntax_non_deferred(intermediate)
        inverted = converter._invert_db_syntax_non_deferred(converted)
        assert _to_canonical(inverted) == _to_canonical(intermediate)


# ── _convert_db_syntax_deferred ────────────────────────────────────────────────


class TestConvertDbSyntaxDeferred:
    """Tests for E/Z isotope → deferred placeholder conversion."""

    def test_produces_pipe_separator(self, converter: CRISPSmiles):
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            "C/C=C/C", smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._convert_db_syntax_deferred(intermediate)
        assert "|" in result

    def test_produces_db_stereo_placeholders(self, converter: CRISPSmiles):
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            "C/C=C/C", smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._convert_db_syntax_deferred(intermediate)
        assert "[DB_STEREO_" in result

    def test_no_isotopes_no_pipe(self, converter: CRISPSmiles):
        result = converter._convert_db_syntax_deferred("CCO")
        assert "|" not in result
        assert result == "CCO"

    def test_ignores_rs_isotopes(self, converter: CRISPSmiles):
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            "C[C@H](O)C(=O)O", smiles_mode="canonical"
        )
        assert intermediate is not None
        result = converter._convert_db_syntax_deferred(intermediate)
        assert "[DB_STEREO_" not in result
        assert "|" not in result


# ── _invert_db_syntax_deferred ─────────────────────────────────────────────────


class TestInvertDbSyntaxDeferred:
    """Tests for deferred E/Z → isotope label inversion."""

    def test_inverts_deferred_ez_to_isotopes(self, converter: CRISPSmiles):
        smiles = "C/C=C/C"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        converted = converter._convert_db_syntax_deferred(intermediate)
        inverted = converter._invert_db_syntax_deferred(converted)
        assert str(STEREO_E_ISOTOPE) in inverted or str(STEREO_Z_ISOTOPE) in inverted

    def test_no_pipe_returns_unchanged(self, converter: CRISPSmiles):
        result = converter._invert_db_syntax_deferred("CCO")
        assert result == "CCO"

    def test_preserves_rs_deferred_assignments(self, converter: CRISPSmiles):
        """E/Z inversion should leave [STEREO_CENTER_N] assignments in the | block."""
        smiles = "C/C=C/[C@H](O)C(=O)O"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        rs_deferred = converter._convert_syntax_deferred(intermediate)
        ez_deferred = converter._convert_db_syntax_deferred(rs_deferred)
        # Now invert only E/Z
        inverted = converter._invert_db_syntax_deferred(ez_deferred)
        # R/S deferred assignments should still be present
        assert "[STEREO_CENTER_" in inverted
        assert "|" in inverted

    def test_roundtrip_invert(self, converter: CRISPSmiles):
        smiles = "C/C=C/C"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        converted = converter._convert_db_syntax_deferred(intermediate)
        inverted = converter._invert_db_syntax_deferred(converted)
        assert _to_canonical(inverted) == _to_canonical(intermediate)


# ── _revert_db_isotopes_to_stereo ──────────────────────────────────────────────


class TestRevertDbIsotopesToStereo:
    """Tests for the E/Z isotope → bond stereo restoration on RDKit Mol."""

    def test_restores_ez_stereo(self, converter: CRISPSmiles):
        smiles = "C/C=C/C"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        mol = Chem.MolFromSmiles(intermediate)
        result_mol = converter._revert_db_isotopes_to_stereo(mol)
        result_smi = Chem.MolToSmiles(result_mol, canonical=True)
        can_orig = _to_canonical(smiles)
        assert result_smi == can_orig

    def test_no_ez_isotopes_unchanged(self, converter: CRISPSmiles):
        mol = Chem.MolFromSmiles("CCO")
        result_mol = converter._revert_db_isotopes_to_stereo(mol)
        result_smi = Chem.MolToSmiles(result_mol, canonical=True)
        assert result_smi == "CCO"

    def test_modifies_mol_in_place(self, converter: CRISPSmiles):
        """_revert_db_isotopes_to_stereo modifies the input mol in-place and returns it."""
        smiles = "C/C=C/C"
        intermediate = converter._generate_local_stereochem_syntax_for_mol(
            smiles, smiles_mode="canonical"
        )
        assert intermediate is not None
        mol = Chem.MolFromSmiles(intermediate)
        result_mol = converter._revert_db_isotopes_to_stereo(mol)
        assert result_mol is mol  # Same object
