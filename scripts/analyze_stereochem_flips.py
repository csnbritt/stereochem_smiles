"""Analyze stereochemistry flips between predicted and target SMILES.

Compares per-atom stereo assignments in predictions vs. targets to
count how often the model "flips" a stereo center (R↔S, E↔Z) rather than
getting it exactly right or dropping it entirely.

For CRISP SMILES inputs, the string is first decoded to standard SMILES
via CRISPSmilesConverter.decode(), then stereo is extracted via RDKit.
"""

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from rdkit import Chem, RDLogger

from crisp_smiles.main import CRISPSmiles

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

_CRISP_CONVERTER = CRISPSmiles()


@dataclass(frozen=True)
class FlipStats:
    """Summary statistics for stereochemistry flips.

    Attributes:
        total_stereo_centers (int): Total stereo centers found in targets.
        exact (int): Centers where prediction matches target exactly.
        flipped (int): Centers where prediction has the opposite assignment.
        dropped (int): Centers where prediction has no stereo assignment.
        added (int): Centers where target has no stereo but prediction does.
    """

    total_stereo_centers: int
    exact: int
    flipped: int
    dropped: int
    added: int


def _extract_stereo_from_smiles(smiles: str) -> dict[int, str]:
    """Extract stereo assignments keyed by atom index from a SMILES string.

    If the input contains CRISP stereo tokens, it is first decoded to
    standard SMILES via CRISPSmilesConverter.decode(), then stereo is
    extracted via RDKit.

    Args:
        smiles (str): A SMILES or CRISP SMILES string.

    Returns:
        Dict[int, str]: Mapping from atom index (0-based) to stereo label
        ("R", "S", "E", or "Z").
    """
    # Decode CRISP SMILES to standard SMILES if stereo tokens are present
    if "[STEREO_" in smiles or "[DB_STEREO_" in smiles or "|" in smiles:
        try:
            decoded = _CRISP_CONVERTER.decode(smiles)
            if decoded is not None:
                smiles = decoded
        except Exception as e:
            logging.getLogger(__name__).debug(
                "CRISP decode failed for '%s': %s", smiles, e
            )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    stereo_map: dict[int, str] = {}

    for atom in mol.GetAtoms():
        chirality = atom.GetChiralTag()
        if chirality == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            stereo_map[atom.GetIdx()] = "R"
        elif chirality == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            stereo_map[atom.GetIdx()] = "S"

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE):
            # Use the lower atom index as the key
            key = min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            stereo_map[key] = "Z" if stereo == Chem.BondStereo.STEREOZ else "E"

    return stereo_map


def analyze_pair(pred: str, target: str) -> FlipStats:
    """Analyze stereochemistry flips for a single prediction/target pair.

    Args:
        pred (str): Predicted SMILES or CRISP SMILES string.
        target (str): Target SMILES or CRISP SMILES string.

    Returns:
        FlipStats: Statistics for this pair.
    """
    pred_stereo = _extract_stereo_from_smiles(pred)
    target_stereo = _extract_stereo_from_smiles(target)

    all_keys = set(pred_stereo) | set(target_stereo)
    total = len(target_stereo)
    exact = 0
    flipped = 0
    dropped = 0
    added = 0

    flip_pairs = {("R", "S"), ("S", "R"), ("E", "Z"), ("Z", "E")}

    for key in all_keys:
        t = target_stereo.get(key)
        p = pred_stereo.get(key)
        if t is not None and p is not None:
            if t == p:
                exact += 1
            elif (t, p) in flip_pairs:
                flipped += 1
            # else: different type entirely (unlikely but counted as mismatch)
        elif t is not None and p is None:
            dropped += 1
        elif t is None and p is not None:
            added += 1

    return FlipStats(
        total_stereo_centers=total,
        exact=exact,
        flipped=flipped,
        dropped=dropped,
        added=added,
    )


def analyze_file(
    generations_path: Path,
    output_path: Path | None = None,
) -> FlipStats:
    """Analyze stereochemistry flips from an eval_generations file.

    The file format is: ``index\\tprediction\\ttarget`` per line.

    Args:
        generations_path (Path): Path to the eval_generations file.
        output_path (Path | None): Optional path to write detailed results.

    Returns:
        FlipStats: Aggregate statistics across all pairs.
    """
    total_exact = 0
    total_flipped = 0
    total_dropped = 0
    total_added = 0
    total_centers = 0
    n_pairs = 0

    flip_counter: Counter[str] = Counter()

    with generations_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pred = parts[1]
            target = parts[2]
            stats = analyze_pair(pred, target)
            total_exact += stats.exact
            total_flipped += stats.flipped
            total_dropped += stats.dropped
            total_added += stats.added
            total_centers += stats.total_stereo_centers
            n_pairs += 1

            if stats.flipped > 0:
                flip_counter["flipped"] += 1
            if (
                stats.exact == stats.total_stereo_centers
                and stats.total_stereo_centers > 0
            ):
                flip_counter["all_exact"] += 1

    aggregate = FlipStats(
        total_stereo_centers=total_centers,
        exact=total_exact,
        flipped=total_flipped,
        dropped=total_dropped,
        added=total_added,
    )

    print(f"\n{'=' * 60}")
    print("Stereochemistry Flip Analysis")
    print("=" * 60)
    print(f"  Pairs analyzed      : {n_pairs}")
    print(f"  Total stereo centers: {total_centers}")
    print(f"  Exact               : {total_exact}")
    print(f"  Flipped             : {total_flipped}")
    print(f"  Dropped             : {total_dropped}")
    print(f"  Added               : {total_added}")
    if total_centers > 0:
        print(f"  Flip rate           : {total_flipped / total_centers:.4f}")
        print(f"  Exact rate          : {total_exact / total_centers:.4f}")
    print(f"  Pairs with flips    : {flip_counter['flipped']}")
    print(f"  Pairs all-exact     : {flip_counter['all_exact']}")
    print(f"{'=' * 60}\n")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write("metric\tvalue\n")
            f.write(f"pairs_analyzed\t{n_pairs}\n")
            f.write(f"total_stereo_centers\t{total_centers}\n")
            f.write(f"exact\t{total_exact}\n")
            f.write(f"flipped\t{total_flipped}\n")
            f.write(f"dropped\t{total_dropped}\n")
            f.write(f"added\t{total_added}\n")
            if total_centers > 0:
                f.write(f"flip_rate\t{total_flipped / total_centers:.6f}\n")
                f.write(f"exact_rate\t{total_exact / total_centers:.6f}\n")
            f.write(f"pairs_with_flips\t{flip_counter['flipped']}\n")
            f.write(f"pairs_all_exact\t{flip_counter['all_exact']}\n")
        print(f"✓ Detailed results saved to {output_path}")

    return aggregate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_stereochem_flips",
        description="Analyze stereochemistry flips between predicted and target SMILES.",
    )
    parser.add_argument(
        "generations_path",
        type=Path,
        help="Path to an eval_generations file (format: idx\\tpred\\ttarget).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write detailed results TSV.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    analyze_file(args.generations_path, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
