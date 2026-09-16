"""
Unit tests for the public encode/decode API of CRISPSmiles.

Covers:
  - Round-trip fidelity (encode → decode → canonical comparison)
  - Deferred vs non-deferred syntax
  - Canonical vs random smiles_mode
  - Molecules with R/S, E/Z, both, and no stereochemistry
  - Invalid SMILES handling
  - Custom stereo tokens
  - Edge cases (empty string, no stereo tokens in decode, etc.)
"""

import pytest
from rdkit import Chem

from crisp_smiles.main import CRISPSmiles

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def converter() -> CRISPSmiles:
    return CRISPSmiles()


@pytest.fixture
def converter_custom_tokens() -> CRISPSmiles:
    return CRISPSmiles(
        stereo_r_token="[R]",
        stereo_s_token="[S]",
        stereo_e_token="[E]",
        stereo_z_token="[Z]",
    )


# ── Molecules with known stereochemistry ─────────────────────────────────────

# (label, canonical SMILES, has_tetrahedral, has_double_bond)
STEREO_MOLECULES = [
    # Tetrahedral only
    ("lactic_acid", "C[C@H](O)C(=O)O", True, False),
    ("alanine", "C[C@H](N)C(=O)O", True, False),
    ("mandelic_acid", "O[C@H](C(=O)O)c1ccccc1", True, False),
    # Double bond only
    ("trans_2_butene", "C/C=C/C", False, True),
    ("cis_2_butene", "C/C=C\\C", False, True),
    ("trans_stilbene", "C(/c1ccccc1)=C/c1ccccc1", False, True),
    # Both tetrahedral and double bond
    ("both_stereo", "C/C=C/[C@H](O)C(=O)O", True, True),
    # No stereochemistry
    ("ethanol", "CCO", False, False),
    ("benzene", "c1ccccc1", False, False),
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O", False, False),
]

# Molecules with multiple stereocenters
MULTI_CENTER_MOLECULES = [
    ("tartaric_acid", "O[C@H](C(=O)O)[C@@H](O)C(=O)O"),  # 2 R/S centers
    ("2_3_butanediol", "C[C@H](O)[C@@H](O)C"),  # 2 R/S centers
]


# ── encode: basic behavior ────────────────────────────────────────────────────


class TestEncodeBasic:
    def test_encode_no_stereo_returns_smiles(self, converter: CRISPSmiles):
        """Molecules without stereo should return canonical SMILES unchanged."""
        result = converter.encode("CCO", deferred=True, smiles_mode="canonical")
        assert result == "CCO"

    def test_encode_no_stereo_non_deferred(self, converter: CRISPSmiles):
        result = converter.encode("CCO", deferred=False, smiles_mode="canonical")
        assert result == "CCO"

    def test_encode_no_stereo_random_mode(self, converter: CRISPSmiles):
        """Random mode on non-stereo should still produce valid SMILES."""
        result = converter.encode("CCO", deferred=True, smiles_mode="random")
        mol = Chem.MolFromSmiles(result)
        assert mol is not None

    def test_encode_returns_str_for_valid_smiles(self, converter: CRISPSmiles):
        result = converter.encode("C[C@H](O)C(=O)O", deferred=True)
        assert isinstance(result, str)

    def test_encode_invalid_smiles_returns_none(self, converter: CRISPSmiles):
        """Invalid SMILES without stereo markers returns None."""
        result = converter.encode("invalid_smiles!!", deferred=True)
        assert result is None

    def test_encode_invalid_smiles_with_at_returns_none(self, converter: CRISPSmiles):
        """Invalid SMILES with @ should return None (not crash with TypeError)."""
        result = converter.encode("C[C@H](O)C(=O)Oinvalid", deferred=True)
        assert result is None

    def test_encode_empty_string(self, converter: CRISPSmiles):
        """Empty string: RDKit parses it as an empty molecule, so encode returns empty string."""
        result = converter.encode("", deferred=True)
        assert result == ""


# ── encode: deferred vs non-deferred ──────────────────────────────────────────


class TestEncodeDeferred:
    @pytest.mark.parametrize("label,smiles,has_rs,has_ez", STEREO_MOLECULES)
    def test_encode_deferred_has_pipe(
        self, converter: CRISPSmiles, label, smiles, has_rs, has_ez
    ):
        """Deferred encoding of stereo molecules should contain | separator."""
        if not has_rs and not has_ez:
            pytest.skip("No stereochemistry to encode")
        result = converter.encode(smiles, deferred=True, smiles_mode="canonical")
        assert result is not None
        assert "|" in result

    @pytest.mark.parametrize("label,smiles,has_rs,has_ez", STEREO_MOLECULES)
    def test_encode_non_deferred_no_pipe(
        self, converter: CRISPSmiles, label, smiles, has_rs, has_ez
    ):
        """Non-deferred encoding should not contain | separator."""
        if not has_rs and not has_ez:
            pytest.skip("No stereochemistry to encode")
        result = converter.encode(smiles, deferred=False, smiles_mode="canonical")
        assert result is not None
        assert "|" not in result

    @pytest.mark.parametrize("label,smiles,has_rs,has_ez", STEREO_MOLECULES)
    def test_encode_non_deferred_has_inline_tokens(
        self, converter: CRISPSmiles, label, smiles, has_rs, has_ez
    ):
        """Non-deferred encoding should contain inline stereo tokens."""
        if not has_rs and not has_ez:
            pytest.skip("No stereochemistry to encode")
        result = converter.encode(smiles, deferred=False, smiles_mode="canonical")
        assert result is not None
        has_token = any(
            t in result
            for t in ["[STEREO_R]", "[STEREO_S]", "[STEREO_E]", "[STEREO_Z]"]
        )
        assert has_token

    @pytest.mark.parametrize("label,smiles,has_rs,has_ez", STEREO_MOLECULES)
    def test_encode_deferred_has_stereo_center_or_db_stereo(
        self, converter: CRISPSmiles, label, smiles, has_rs, has_ez
    ):
        """Deferred encoding should contain [STEREO_CENTER_N] or [DB_STEREO_N]."""
        if not has_rs and not has_ez:
            pytest.skip("No stereochemistry to encode")
        result = converter.encode(smiles, deferred=True, smiles_mode="canonical")
        assert result is not None
        has_placeholder = "[STEREO_CENTER_" in result or "[DB_STEREO_" in result
        assert has_placeholder


# ── encode: canonical vs random ───────────────────────────────────────────────


class TestEncodeSmilesMode:
    def test_encode_random_different_from_canonical(self, converter: CRISPSmiles):
        """Random mode should usually produce different output from canonical."""
        smiles = "C[C@H](O)C(=O)O"
        can = converter.encode(smiles, smiles_mode="canonical")
        rand = converter.encode(smiles, smiles_mode="random")
        assert can is not None
        assert rand is not None
        # They might occasionally be the same for simple molecules,
        # but for lactic acid they should differ
        assert can != rand

    def test_encode_invalid_smiles_mode_falls_back_to_canonical(
        self, converter: CRISPSmiles
    ):
        """An unrecognized smiles_mode should fall back to canonical."""
        smiles = "C[C@H](O)C(=O)O"
        result = converter.encode(smiles, smiles_mode="invalid_mode")
        can = converter.encode(smiles, smiles_mode="canonical")
        assert result is not None
        assert can is not None
        assert result == can

    def test_encode_random_produces_valid_smiles(self, converter: CRISPSmiles):
        """Random mode output should be parseable by RDKit after decode."""
        smiles = "C/C=C/[C@H](O)C(=O)O"
        encoded = converter.encode(smiles, smiles_mode="random")
        assert encoded is not None
        decoded = converter.decode(encoded, deferred=True)
        mol = Chem.MolFromSmiles(decoded)
        assert mol is not None


# ── encode: custom tokens ─────────────────────────────────────────────────────


class TestEncodeCustomTokens:
    def test_custom_tokens_in_non_deferred_output(
        self, converter_custom_tokens: CRISPSmiles
    ):
        result = converter_custom_tokens.encode(
            "C[C@H](O)C(=O)O", deferred=False, smiles_mode="canonical"
        )
        assert result is not None
        assert "[R]" in result or "[S]" in result

    def test_custom_tokens_in_deferred_output(
        self, converter_custom_tokens: CRISPSmiles
    ):
        result = converter_custom_tokens.encode(
            "C[C@H](O)C(=O)O", deferred=True, smiles_mode="canonical"
        )
        assert result is not None
        assert "[R]" in result or "[S]" in result

    @pytest.mark.xfail(
        reason="Bug: _invert_syntax_deferred and _invert_db_syntax_deferred use hardcoded "
        "[STEREO_R]/[STEREO_S]/[STEREO_E]/[STEREO_Z] in regex instead of self._stereo_*_token, "
        "so custom tokens break the decode path"
    )
    def test_custom_tokens_decode_roundtrip(self, converter_custom_tokens: CRISPSmiles):
        smiles = "C[C@H](O)C(=O)O"
        encoded = converter_custom_tokens.encode(
            smiles, deferred=True, smiles_mode="canonical"
        )
        assert encoded is not None
        decoded = converter_custom_tokens.decode(encoded, deferred=True)
        can_orig = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        can_dec = Chem.MolToSmiles(Chem.MolFromSmiles(decoded), canonical=True)
        assert can_orig == can_dec


# ── decode: basic behavior ────────────────────────────────────────────────────


class TestDecodeBasic:
    def test_decode_no_stereo_tokens_returns_input(self, converter: CRISPSmiles):
        """If no stereo tokens are present, decode returns the input unchanged."""
        result = converter.decode("CCO", deferred=True)
        assert result == "CCO"

    def test_decode_no_stereo_tokens_non_deferred(self, converter: CRISPSmiles):
        result = converter.decode("CCO", deferred=False)
        assert result == "CCO"

    def test_decode_empty_string(self, converter: CRISPSmiles):
        result = converter.decode("", deferred=True)
        assert result == ""


# ── Round-trip: encode → decode → canonical comparison ─────────────────────────


class TestRoundTrip:
    @pytest.mark.parametrize("label,smiles,has_rs,has_ez", STEREO_MOLECULES)
    @pytest.mark.parametrize("deferred", [True, False])
    def test_roundtrip_canonical(
        self, converter: CRISPSmiles, label, smiles, has_rs, has_ez, deferred
    ):
        """encode → decode should preserve the molecule's canonical SMILES."""
        encoded = converter.encode(smiles, deferred=deferred, smiles_mode="canonical")
        assert encoded is not None, f"encode returned None for {label}"
        decoded = converter.decode(encoded, deferred=deferred)
        can_orig = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        can_dec = Chem.MolToSmiles(Chem.MolFromSmiles(decoded), canonical=True)
        assert can_orig == can_dec, f"Round-trip failed for {label}"

    @pytest.mark.parametrize("label,smiles,has_rs,has_ez", STEREO_MOLECULES)
    @pytest.mark.parametrize("deferred", [True, False])
    def test_roundtrip_random(
        self, converter: CRISPSmiles, label, smiles, has_rs, has_ez, deferred
    ):
        """Random mode encode → decode should preserve canonical SMILES."""
        encoded = converter.encode(smiles, deferred=deferred, smiles_mode="random")
        assert encoded is not None, f"encode returned None for {label}"
        decoded = converter.decode(encoded, deferred=deferred)
        can_orig = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        can_dec = Chem.MolToSmiles(Chem.MolFromSmiles(decoded), canonical=True)
        assert can_orig == can_dec, f"Random round-trip failed for {label}"

    @pytest.mark.parametrize("label,smiles", MULTI_CENTER_MOLECULES)
    @pytest.mark.parametrize("deferred", [True, False])
    def test_roundtrip_multi_center(
        self, converter: CRISPSmiles, label, smiles, deferred
    ):
        """Multi-center molecules should round-trip correctly."""
        encoded = converter.encode(smiles, deferred=deferred, smiles_mode="canonical")
        assert encoded is not None
        decoded = converter.decode(encoded, deferred=deferred)
        can_orig = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        can_dec = Chem.MolToSmiles(Chem.MolFromSmiles(decoded), canonical=True)
        assert can_orig == can_dec, f"Multi-center round-trip failed for {label}"

    @pytest.mark.parametrize("deferred", [True, False])
    def test_roundtrip_random_stereo_preserved(self, converter: CRISPSmiles, deferred):
        """Multiple random encodings of the same molecule should decode to the same canonical."""
        smiles = "C/C=C/[C@H](O)C(=O)O"
        can_orig = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        for _ in range(5):
            encoded = converter.encode(smiles, deferred=deferred, smiles_mode="random")
            assert encoded is not None
            decoded = converter.decode(encoded, deferred=deferred)
            can_dec = Chem.MolToSmiles(Chem.MolFromSmiles(decoded), canonical=True)
            assert can_orig == can_dec


# ── encode: stereo token count consistency ────────────────────────────────────


class TestStereoTokenConsistency:
    @pytest.mark.parametrize("label,smiles,has_rs,has_ez", STEREO_MOLECULES)
    def test_deferred_and_non_deferred_same_stereo_count(
        self, converter: CRISPSmiles, label, smiles, has_rs, has_ez
    ):
        """Both deferred and non-deferred should encode the same number of stereocenters."""
        if not has_rs and not has_ez:
            pytest.skip("No stereochemistry")
        deferred_result = converter.encode(
            smiles, deferred=True, smiles_mode="canonical"
        )
        nondef_result = converter.encode(
            smiles, deferred=False, smiles_mode="canonical"
        )
        assert deferred_result is not None
        assert nondef_result is not None

        # Count stereo assignments in deferred (via | block)
        deferred_count = (
            deferred_result.count("[STEREO_R]")
            + deferred_result.count("[STEREO_S]")
            + deferred_result.count("[STEREO_E]")
            + deferred_result.count("[STEREO_Z]")
        )
        nondef_count = (
            nondef_result.count("[STEREO_R]")
            + nondef_result.count("[STEREO_S]")
            + nondef_result.count("[STEREO_E]")
            + nondef_result.count("[STEREO_Z]")
        )
        assert deferred_count == nondef_count

    def test_random_mode_preserves_stereo_count(self, converter: CRISPSmiles):
        """Multiple random encodings should have the same stereo token count."""
        smiles = "C/C=C/[C@H](O)C(=O)O"
        can_result = converter.encode(smiles, deferred=True, smiles_mode="canonical")
        assert can_result is not None
        can_count = (
            can_result.count("[STEREO_R]")
            + can_result.count("[STEREO_S]")
            + can_result.count("[STEREO_E]")
            + can_result.count("[STEREO_Z]")
        )
        for _ in range(5):
            rand_result = converter.encode(smiles, deferred=True, smiles_mode="random")
            assert rand_result is not None
            rand_count = (
                rand_result.count("[STEREO_R]")
                + rand_result.count("[STEREO_S]")
                + rand_result.count("[STEREO_E]")
                + rand_result.count("[STEREO_Z]")
            )
            assert rand_count == can_count
