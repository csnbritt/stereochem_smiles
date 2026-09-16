"""
Unit tests for the SMILES vocabulary.

Covers:
  - New E/Z and DB_STEREO tokens exist with correct IDs
  - Utility tokens are shifted correctly after the new stereo tokens
  - All token IDs are unique and contiguous
  - Backslash token is a single backslash character
"""

from models.tokenizer.vocab import smiles_token_to_id_dict


def test_stereo_e_token_exists():
    """[STEREO_E] should be present in the vocabulary."""
    assert "[STEREO_E]" in smiles_token_to_id_dict


def test_stereo_z_token_exists():
    """[STEREO_Z] should be present in the vocabulary."""
    assert "[STEREO_Z]" in smiles_token_to_id_dict


def test_db_stereo_tokens_exist():
    """[DB_STEREO_1] through [DB_STEREO_25] should all be present."""
    for i in range(1, 26):
        token = f"[DB_STEREO_{i}]"
        assert token in smiles_token_to_id_dict, f"Missing token: {token}"


def test_stereo_e_z_ids_are_sequential():
    """[STEREO_E] and [STEREO_Z] should have consecutive IDs."""
    e_id = smiles_token_to_id_dict["[STEREO_E]"]
    z_id = smiles_token_to_id_dict["[STEREO_Z]"]
    assert z_id == e_id + 1


def test_db_stereo_ids_are_sequential():
    """[DB_STEREO_1] through [DB_STEREO_25] should have consecutive IDs."""
    first_id = smiles_token_to_id_dict["[DB_STEREO_1]"]
    for i in range(1, 26):
        token = f"[DB_STEREO_{i}]"
        assert smiles_token_to_id_dict[token] == first_id + (i - 1)


def test_stereo_e_before_db_stereo():
    """[STEREO_E] should come before [DB_STEREO_1]."""
    assert (
        smiles_token_to_id_dict["[STEREO_E]"] < smiles_token_to_id_dict["[DB_STEREO_1]"]
    )


def test_db_stereo_before_utility_tokens():
    """[DB_STEREO_25] should come before [UTILITY_TOKEN_26]."""
    db_last = smiles_token_to_id_dict["[DB_STEREO_25]"]
    util_first = smiles_token_to_id_dict["[UTILITY_TOKEN_26]"]
    assert db_last < util_first


def test_utility_token_26_id():
    """[UTILITY_TOKEN_26] should be at ID 874 (shifted by 27 from original 847)."""
    assert smiles_token_to_id_dict["[UTILITY_TOKEN_26]"] == 874


def test_all_ids_unique():
    """All token IDs should be unique."""
    ids = list(smiles_token_to_id_dict.values())
    assert len(ids) == len(set(ids)), "Duplicate token IDs found"


def test_backslash_token_is_single_backslash():
    """The backslash entry should be a single backslash character, not double-escaped."""
    assert "\\" in smiles_token_to_id_dict
    assert smiles_token_to_id_dict["\\"] == 31


def test_backslash_token_not_double_escaped():
    """The double-backslash should NOT be a key in the vocabulary."""
    assert "\\\\" not in smiles_token_to_id_dict


def test_vocab_size_increased():
    """Vocabulary should have grown by 27 (2 E/Z + 25 DB_STEREO) tokens."""
    # Original size was 1024 tokens; now should be 1024 + 27 = 1051
    assert len(smiles_token_to_id_dict) == 1051
