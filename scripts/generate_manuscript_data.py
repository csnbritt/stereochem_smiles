import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem


@dataclass(frozen=True)
class ExperimentSpec:
    dataset: str
    task: str


def _read_lines(path: Path) -> List[str]:
    """
    Read a text file and return stripped, non-empty lines.

    Args:
        path (Path): Path to a text file.

    Returns:
        List[str]: Stripped lines.
    """
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_smiles_as_mols(smiles_path: Path) -> List[Chem.Mol]:
    """
    Load SMILES from a newline-delimited file and parse them into RDKit molecules.

    Args:
        smiles_path (Path): Path to a SMILES text file (one SMILES per line).

    Returns:
        List[Chem.Mol]: Parsed molecules (invalid SMILES are skipped).
    """
    mols: List[Chem.Mol] = []
    for smi in _read_lines(smiles_path):
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            mol = None
        if mol is not None:
            mols.append(mol)
    return mols


def split_train_val(
    mols: Sequence[Chem.Mol],
    train_size: int,
) -> Tuple[List[Chem.Mol], List[Chem.Mol]]:
    """
    Split molecules into train/validation partitions.

    Args:
        mols (Sequence[Chem.Mol]): Input molecules.
        train_size (int): Number of molecules to use for training.

    Returns:
        Tuple[List[Chem.Mol], List[Chem.Mol]]:
            - Train molecules.
            - Validation molecules.
    """
    train = list(mols[:train_size])
    val = list(mols[train_size:])
    return train, val


def get_default_experiments() -> List[ExperimentSpec]:
    """
    Return the default experiment matrix.

    Returns:
        List[ExperimentSpec]: Experiment specs.
    """
    return [
        ExperimentSpec("zinc", "random_smiles_to_canonical_smiles"),
        ExperimentSpec("zinc", "random_crisp_nondef_to_canonical_crisp_nondef"),
        ExperimentSpec("zinc", "random_crisp_def_to_canonical_crisp_def"),
        ExperimentSpec("zinc", "random_crisp_def_to_canonical_smiles"),
        ExperimentSpec("zinc", "random_smiles_to_canonical_crisp_def"),
        ExperimentSpec("USPTO_STEREO", "random_smiles_to_canonical_smiles"),
        ExperimentSpec("USPTO_STEREO", "random_crisp_nondef_to_canonical_crisp_nondef"),
        ExperimentSpec("USPTO_STEREO", "random_crisp_def_to_canonical_crisp_def"),
    ]


def run_experiment(
    experiment: ExperimentSpec,
    train_mols: Sequence[Chem.Mol],
    val_mols: Sequence[Chem.Mol],
    *,
    mode: str,
) -> None:
    """
    Run a single experiment.

    Args:
        experiment (ExperimentSpec): Which dataset/task combination to run.
        train_mols (Sequence[Chem.Mol]): Training molecules.
        val_mols (Sequence[Chem.Mol]): Validation molecules.
        mode (str): Preprocessing/training mode string.

    Raises:
        RuntimeError: If the training components referenced by the manuscript script
            are not implemented/importable in this repo yet.
    """
    try:
        from models.training.model import (
            MODE_RANDOM_TO_CANONICAL_NEW_SYNTAX,  # noqa: WPS433
        )
    except Exception as e:
        raise RuntimeError(
            "Could not import training mode constants from `models.training.model`. "
            "Make sure your training module is available before running this script."
        ) from e

    _ = MODE_RANDOM_TO_CANONICAL_NEW_SYNTAX

    raise RuntimeError(
        "Training scaffold only: wire in your CustomSmallConfig/CustomTokenizer/SmilesTrainer "
        "implementation here."
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="generate_manuscript_data")

    parser.add_argument(
        "--zinc-smiles-path",
        type=Path,
        default=Path("/content/zinc250k.txt"),
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=245000,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="random_to_canonical_new_syntax",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    experiments = get_default_experiments()

    zinc_mols = load_smiles_as_mols(args.zinc_smiles_path)
    train_mols, val_mols = split_train_val(zinc_mols, train_size=args.train_size)

    for exp in experiments:
        if exp.dataset == "zinc":
            dataset_train = train_mols
            dataset_val = val_mols
        elif exp.dataset == "USPTO_STEREO":
            dataset_train = train_mols
            dataset_val = val_mols
        else:
            raise ValueError(f"Unknown dataset '{exp.dataset}'")

        run_experiment(
            exp,
            dataset_train,
            dataset_val,
            mode=args.mode,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
