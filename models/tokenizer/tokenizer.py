import json
import os
import re
from typing import Any, Dict, List, Tuple

from transformers import (
    PreTrainedTokenizer,
)


class CustomTokenizer(PreTrainedTokenizer):
    """
    A tokenizer built from a user-supplied token -> integer mapping.
    Supports both character-level and multi-character tokens (e.g. Br, Cl, >>).

    Tokenization strategy:
        - If the text contains spaces, it is assumed to be a pre-tokenized
          string (e.g. output of preprocess_sentence_template) and will be
          split on whitespace. Spaces themselves are not tokens.
        - If the text contains no spaces, it is tokenized character by
          character, with unknown characters replaced by unk_token.

    Args:
        token_to_id: A dictionary mapping tokens (str) to integer IDs.
        id_to_token: A dictionary mapping integer IDs to tokens (str).
                     If not provided, it will be derived from token_to_id.
        unk_token:   The unknown token string. Must exist in token_to_id.
        pad_token:   The padding token string. Must exist in token_to_id.
        mask_token:  The mask token string. Must exist in token_to_id.
        cls_token:   The CLS token string. Must exist in token_to_id.
        sep_token:   The SEP token string. Must exist in token_to_id.
    """

    def __init__(
        self,
        token_to_id: Dict[str, int],
        id_to_token: Dict[int, str] | None = None,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        mask_token: str = "[MASK]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        bos_token: str = "^",
        eos_token: str = "$",
        **kwargs,
    ):
        special_tokens = {
            "unk_token": unk_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "cls_token": cls_token,
            "sep_token": sep_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
        }
        for name, token in special_tokens.items():
            if token not in token_to_id:
                raise ValueError(
                    f"Special token '{token}' ({name}) not found in token_to_id. "
                    f"Please include it in your vocabulary."
                )

        self._token_to_id = token_to_id
        self._id_to_token = id_to_token or {v: k for k, v in token_to_id.items()}

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            cls_token=cls_token,
            sep_token=sep_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def get_vocab(self) -> Dict[str, int]:
        return self._token_to_id.copy()

    def _tokenize(self, text: Any, **kwargs) -> List[str]:
        """
        Tokenize a string.

        If the string contains spaces (i.e. it has been preprocessed by
        preprocess_sentence_template), split on whitespace and treat each
        element as a token. Spaces are discarded and are never tokens.

        If the string contains no spaces, fall back to character-level
        tokenization for raw/unprocessed input.

        Unknown tokens are replaced with unk_token.
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string.")

        text = self.preprocess_sentence_reaction_smiles(text)

        if " " in text:
            # Pre-tokenized path: split on whitespace, discard empty strings
            raw_tokens = text.split()
        else:
            # Character-level fallback for raw strings
            raw_tokens = list(text)

        return [
            token if token in self._token_to_id else self.unk_token
            for token in raw_tokens
        ]

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        result = self._id_to_token.get(index, self.unk_token)
        return result if result is not None else (self.unk_token or "")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Rejoin tokens back into a readable string.
        Tokens are joined with a space so the output mirrors the
        preprocessed SMILES format.
        """
        return " ".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: List[int] | None = None
    ) -> List[int]:
        """
        Do not add any special tokens (CLS, SEP) around the sequence.
        Overrides the default BERT-style behaviour of prepending [CLS]
        and appending [SEP].
        """
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: List[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Returns a mask of 0s only, since we add no special tokens.
        """
        if token_ids_1 is None:
            return [0] * len(token_ids_0)
        return [0] * (len(token_ids_0) + len(token_ids_1))

    def preprocess_sentence_reaction_smiles(self, reaction_smiles: str) -> str:
        REGEXPS_RXN = {
            "brackets": re.compile(r"(\[[^\]]*\])"),
            "2_ring_nums": re.compile(r"(%\d{2})"),
            "rxn_symbol": re.compile(r"(>>)"),
            "arrow_forward": re.compile(r"(->)"),
            "arrow_backward": re.compile(r"(<-)"),
            "brcl": re.compile(r"(Li|Na|Mg|Si|Ca|Cu|Ag|Pb|Br|Cl|>>)"),
        }

        REGEXP_ORDER_RXN = [
            "brackets",
            "2_ring_nums",
            "arrow_forward",
            "arrow_backward",
            "brcl",
            "rxn_symbol",
        ]

        def split_by(reaction_smiles, regexps):
            if not regexps:
                return list(reaction_smiles)
            regexp = REGEXPS_RXN[regexps[0]]
            splitted = regexp.split(reaction_smiles)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(reaction_smiles, REGEXP_ORDER_RXN)
        string = ""
        for ele in tokens:
            string += ele + " "
        string = "^ " + string + "$"
        return string

    def decode_id_list(self, ids: List[int]) -> str:
        """Convenience method to decode a list of IDs back to a string."""
        tokens = [self._convert_id_to_token(i) for i in ids]
        return self.convert_tokens_to_string(tokens)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> Tuple[str]:
        """
        Save the vocabulary (token -> id mapping) to a JSON file.

        Required by PreTrainedTokenizerBase.save_pretrained, which calls this
        method to write the vocab file alongside tokenizer_config.json.

        Args:
            save_directory (str): Directory in which to save the vocabulary.
            filename_prefix (str | None): Optional prefix prepended to the
                filename (used by save_pretrained for checkpoint subfolders).

        Returns:
            Tuple[str]: A single-element tuple with the path to the saved
                vocabulary file.
        """
        os.makedirs(save_directory, exist_ok=True)

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )

        sorted_vocab = dict(
            sorted(self._token_to_id.items(), key=lambda kv: kv[1])
        )
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(sorted_vocab, f, indent=2, ensure_ascii=False)

        return (vocab_file,)
