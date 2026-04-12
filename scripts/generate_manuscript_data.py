import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from models.tokenizer.tokenizer import CustomTokenizer
from models.tokenizer.vocab import smiles_token_to_id_dict
from models.training.trainer import (
    CustomSmallConfig,
    SmilesPreprocessing,
    SmilesTrainer,
    generate_training_data,
)
from scripts.experiment_definitions import (
    uspto_random_crisp_def_to_canonical_crisp_def,
    uspto_random_crisp_nondef_to_canonical_crisp_nondef,
    uspto_random_smiles_to_canonical_smiles,
    zinc_random_crisp_def_to_canonical_crisp_def,
    zinc_random_crisp_def_to_canonical_smiles,
    zinc_random_crisp_def_to_random_crisp_def,
    zinc_random_crisp_nondef_to_canonical_crisp_nondef,
    zinc_random_crisp_nondef_to_random_crisp_nondef,
    zinc_random_smiles_to_canonical_crisp_def,
    zinc_random_smiles_to_canonical_smiles_config,
)

# ── Reusable preprocessing fragments ─────────────────────────────────────────

RANDOM_SMILES: SmilesPreprocessing = {
    "input_smiles_syntax": "smiles",
    "input_smiles_type": "random",
    "crisp_input_deferred": False,
}
CANONICAL_SMILES: SmilesPreprocessing = {
    "input_smiles_syntax": "smiles",
    "input_smiles_type": "canonical",
    "crisp_input_deferred": False,
}
RANDOM_CRISP_NONDEF: SmilesPreprocessing = {
    "input_smiles_syntax": "crisp_smiles",
    "input_smiles_type": "random",
    "crisp_input_deferred": False,
}
CANONICAL_CRISP_NONDEF: SmilesPreprocessing = {
    "input_smiles_syntax": "crisp_smiles",
    "input_smiles_type": "canonical",
    "crisp_input_deferred": False,
}
RANDOM_CRISP_DEF: SmilesPreprocessing = {
    "input_smiles_syntax": "crisp_smiles",
    "input_smiles_type": "random",
    "crisp_input_deferred": True,
}
CANONICAL_CRISP_DEF: SmilesPreprocessing = {
    "input_smiles_syntax": "crisp_smiles",
    "input_smiles_type": "canonical",
    "crisp_input_deferred": True,
}


# ── ExperimentSpec ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExperimentSpec:
    """
    Specification for a single training experiment.

    Args:
        dataset (str): Dataset identifier ("zinc" or "USPTO_STEREO").
        task (str): Human-readable task name used for logging and directory naming.
        config (CustomSmallConfig): Full model / training configuration for this experiment.
        input_preprocessing (SmilesPreprocessing): How to preprocess the input side.
        output_preprocessing (SmilesPreprocessing): How to preprocess the output side.
    """

    dataset: str
    task: str
    config: CustomSmallConfig
    input_preprocessing: SmilesPreprocessing
    output_preprocessing: SmilesPreprocessing


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


def get_default_experiments() -> List[ExperimentSpec]:
    """
    Return the default experiment matrix.

    Returns:
        List[ExperimentSpec]: Experiment specs for all manuscript experiments.
    """
    return [
        # ── ZINC translation experiments ──────────────────────────────────
        ExperimentSpec(
            dataset="zinc",
            task="random_smiles_to_canonical_smiles",
            config=zinc_random_smiles_to_canonical_smiles_config,
            input_preprocessing=RANDOM_SMILES,
            output_preprocessing=CANONICAL_SMILES,
        ),
        ExperimentSpec(
            dataset="zinc",
            task="random_crisp_nondef_to_canonical_crisp_nondef",
            config=zinc_random_crisp_nondef_to_canonical_crisp_nondef,
            input_preprocessing=RANDOM_CRISP_NONDEF,
            output_preprocessing=CANONICAL_CRISP_NONDEF,
        ),
        ExperimentSpec(
            dataset="zinc",
            task="random_crisp_def_to_canonical_crisp_def",
            config=zinc_random_crisp_def_to_canonical_crisp_def,
            input_preprocessing=RANDOM_CRISP_DEF,
            output_preprocessing=CANONICAL_CRISP_DEF,
        ),
        ExperimentSpec(
            dataset="zinc",
            task="random_crisp_def_to_canonical_smiles",
            config=zinc_random_crisp_def_to_canonical_smiles,
            input_preprocessing=RANDOM_CRISP_DEF,
            output_preprocessing=CANONICAL_SMILES,
        ),
        ExperimentSpec(
            dataset="zinc",
            task="random_smiles_to_canonical_crisp_def",
            config=zinc_random_smiles_to_canonical_crisp_def,
            input_preprocessing=RANDOM_SMILES,
            output_preprocessing=CANONICAL_CRISP_DEF,
        ),
        ExperimentSpec(
            dataset="zinc",
            task="random_crisp_def_to_random_crisp_def",
            config=zinc_random_crisp_def_to_random_crisp_def,
            input_preprocessing=RANDOM_CRISP_DEF,
            output_preprocessing=RANDOM_CRISP_DEF,
        ),
        ExperimentSpec(
            dataset="zinc",
            task="random_crisp_nondef_to_random_crisp_nondef",
            config=zinc_random_crisp_nondef_to_random_crisp_nondef,
            input_preprocessing=RANDOM_CRISP_NONDEF,
            output_preprocessing=RANDOM_CRISP_NONDEF,
        ),
        # ── USPTO_STEREO reaction experiments ─────────────────────────────
        ExperimentSpec(
            dataset="USPTO_STEREO",
            task="random_smiles_to_canonical_smiles",
            config=uspto_random_smiles_to_canonical_smiles,
            input_preprocessing=RANDOM_SMILES,
            output_preprocessing=CANONICAL_SMILES,
        ),
        ExperimentSpec(
            dataset="USPTO_STEREO",
            task="random_crisp_nondef_to_canonical_crisp_nondef",
            config=uspto_random_crisp_nondef_to_canonical_crisp_nondef,
            input_preprocessing=RANDOM_CRISP_NONDEF,
            output_preprocessing=CANONICAL_CRISP_NONDEF,
        ),
        ExperimentSpec(
            dataset="USPTO_STEREO",
            task="random_crisp_def_to_canonical_crisp_def",
            config=uspto_random_crisp_def_to_canonical_crisp_def,
            input_preprocessing=RANDOM_CRISP_DEF,
            output_preprocessing=CANONICAL_CRISP_DEF,
        ),
    ]


def run_experiment(
    experiment: ExperimentSpec,
    train_data: List[str],
    val_data: List[str],
    *,
    test_data: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run a single training experiment end-to-end.

    Uses the config and preprocessing definitions carried by the ExperimentSpec
    directly, builds a tokenizer, trains with SmilesTrainer, and optionally
    evaluates on test data.

    Args:
        experiment (ExperimentSpec): Experiment specification (includes config,
            input/output preprocessing, dataset, and task name).
        train_data (List[str]): Raw training strings (SMILES for translation,
            reaction SMILES for reaction).
        val_data (List[str]): Raw validation strings (same format as train_data).
        test_data (Optional[List[str]]): Raw test strings. If provided, the trained
            model is evaluated and results are saved.

    Returns:
        Dict[str, Any]: Test evaluation results (empty dict if no test data).
    """
    config = experiment.config
    input_preprocessing = experiment.input_preprocessing
    output_preprocessing = experiment.output_preprocessing
    mode = "reaction" if experiment.dataset == "USPTO_STEREO" else "translation"

    tokenizer = CustomTokenizer(token_to_id=smiles_token_to_id_dict)

    # Pre-process test data once (static, no augmentation)
    processed_test = None
    if test_data:
        processed_test = generate_training_data(
            test_data,
            mode=mode,
            input_smiles_preprocessing=input_preprocessing,
            output_smiles_preprocessing=output_preprocessing,
        )

    trainer = SmilesTrainer(
        config=config,
        tokenizer=tokenizer,
        train_data=train_data,
        mode=mode,
        val_data=val_data,
        test_data=processed_test,
        input_smiles_preprocessing=input_preprocessing,
        output_smiles_preprocessing=output_preprocessing,
    )

    trainer.train()

    # Evaluate on test set if available
    results: Dict[str, Any] = {}
    if processed_test:
        test_inputs = [pair[0] for pair in processed_test]
        test_targets = [pair[1] for pair in processed_test]
        final_model_path = str(Path(config.output_dir) / "final_model")
        results = trainer.evaluate_predictions(
            test_inputs,
            test_targets,
            model_path=final_model_path,
        )

        results_path = Path(config.output_dir) / "test_results.txt"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as f:
            f.write(f"exact_match\t{results.get('exact_match', '')}\n")
            f.write(f"token_accuracy\t{results.get('token_accuracy', '')}\n")

    return results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate_manuscript_data",
        description="Run all manuscript training experiments.",
    )

    parser.add_argument(
        "--zinc-smiles-path",
        type=Path,
        default=Path("/data/zinc250k.txt"),
        help="Path to the ZINC 250K SMILES file.",
    )
    parser.add_argument(
        "--uspto-dir",
        type=Path,
        default=Path("/data"),
        help="Directory containing JIN_USPTO_1product_{train,val,test}.txt files.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=245000,
        help="Number of ZINC molecules to use for training (rest go to validation).",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    experiments = get_default_experiments()

    # ── Load ZINC data (SMILES strings, split into train/val) ─────────────
    print("Loading ZINC dataset...")
    zinc_smiles = _read_lines(args.zinc_smiles_path)
    zinc_train = zinc_smiles[: args.train_size]
    zinc_val = zinc_smiles[args.train_size :]
    print(f"  ZINC train: {len(zinc_train)}  |  val: {len(zinc_val)}")

    # ── Load USPTO data (pre-split reaction SMILES) ──────────────────────
    uspto_train_path = args.uspto_dir / "JIN_USPTO_1product_train.txt"
    uspto_val_path = args.uspto_dir / "JIN_USPTO_1product_val.txt"
    uspto_test_path = args.uspto_dir / "JIN_USPTO_1product_test.txt"

    print("Loading USPTO_STEREO dataset...")
    uspto_train = _read_lines(uspto_train_path)
    uspto_val = _read_lines(uspto_val_path)
    uspto_test = _read_lines(uspto_test_path)
    print(
        f"  USPTO train: {len(uspto_train)}  |  val: {len(uspto_val)}"
        f"  |  test: {len(uspto_test)}"
    )

    # ── Run experiments ───────────────────────────────────────────────────
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT {i}/{len(experiments)}: {exp.dataset} / {exp.task}")
        print(f"{'=' * 80}\n")

        if exp.dataset == "zinc":
            run_experiment(exp, zinc_train, zinc_val)
        elif exp.dataset == "USPTO_STEREO":
            run_experiment(exp, uspto_train, uspto_val, test_data=uspto_test)
        else:
            raise ValueError(f"Unknown dataset '{exp.dataset}'")

    print(f"\n{'=' * 80}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
