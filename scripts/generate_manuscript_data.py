import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiment_definitions import (
    uspto_crisp_def_to_crisp_def,
    uspto_crisp_def_to_smiles,
    uspto_crisp_nondef_to_crisp_nondef,
    uspto_smiles_no_stereo_to_smiles_no_stereo,
    uspto_smiles_to_crisp_def,
    uspto_smiles_to_smiles,
    zinc_crisp_def_to_crisp_def,
    zinc_crisp_def_to_smiles,
    zinc_crisp_nondef_to_crisp_nondef,
    zinc_smiles_no_stereo_to_smiles_no_stereo,
    zinc_smiles_to_crisp_def,
    zinc_smiles_to_smiles,
)

from models.tokenizer.tokenizer import CustomTokenizer
from models.tokenizer.vocab import smiles_token_to_id_dict
from models.training.trainer import (
    CustomSmallConfig,
    SmilesPreprocessing,
    SmilesTrainer,
    compute_top_tokens,
    generate_training_data,
)

# ── Reusable preprocessing fragments ─────────────────────────────────────────

RANDOM_SMILES: SmilesPreprocessing = {
    "input_smiles_syntax": "smiles",
    "input_smiles_type": "random",
    "crisp_input_deferred": False,
    "strip_stereo": False,
}
CANONICAL_SMILES: SmilesPreprocessing = {
    "input_smiles_syntax": "smiles",
    "input_smiles_type": "canonical",
    "crisp_input_deferred": False,
    "strip_stereo": False,
}
RANDOM_CRISP_NONDEF: SmilesPreprocessing = {
    "input_smiles_syntax": "crisp_smiles",
    "input_smiles_type": "random",
    "crisp_input_deferred": False,
    "strip_stereo": False,
}
CANONICAL_CRISP_NONDEF: SmilesPreprocessing = {
    "input_smiles_syntax": "crisp_smiles",
    "input_smiles_type": "canonical",
    "crisp_input_deferred": False,
    "strip_stereo": False,
}
RANDOM_CRISP_DEF: SmilesPreprocessing = {
    "input_smiles_syntax": "crisp_smiles",
    "input_smiles_type": "random",
    "crisp_input_deferred": True,
    "strip_stereo": False,
}
CANONICAL_CRISP_DEF: SmilesPreprocessing = {
    "input_smiles_syntax": "crisp_smiles",
    "input_smiles_type": "canonical",
    "crisp_input_deferred": True,
    "strip_stereo": False,
}
STRIPPED_RANDOM_SMILES: SmilesPreprocessing = {
    "input_smiles_syntax": "smiles",
    "input_smiles_type": "random",
    "crisp_input_deferred": False,
    "strip_stereo": True,
}
STRIPPED_CANONICAL_SMILES: SmilesPreprocessing = {
    "input_smiles_syntax": "smiles",
    "input_smiles_type": "canonical",
    "crisp_input_deferred": False,
    "strip_stereo": True,
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


def _read_lines(path: Path) -> list[str]:
    """
    Read a text file and return stripped, non-empty lines.

    Args:
        path (Path): Path to a text file.

    Returns:
        List[str]: Stripped lines.
    """
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_default_experiments() -> list[ExperimentSpec]:
    """
    Return the default experiment matrix.

    Returns:
        List[ExperimentSpec]: Experiment specs for all 12 manuscript experiments.
    """
    return [
        # ── ZINC translation experiments (6 conditions) ───────────────────
        ExperimentSpec(
            dataset="zinc",
            task="smiles_to_smiles",
            config=zinc_smiles_to_smiles,
            input_preprocessing=RANDOM_SMILES,
            output_preprocessing=CANONICAL_SMILES,
        ),
        ExperimentSpec(
            dataset="zinc",
            task="crisp_def_to_crisp_def",
            config=zinc_crisp_def_to_crisp_def,
            input_preprocessing=RANDOM_CRISP_DEF,
            output_preprocessing=CANONICAL_CRISP_DEF,
        ),
        ExperimentSpec(
            dataset="zinc",
            task="crisp_nondef_to_crisp_nondef",
            config=zinc_crisp_nondef_to_crisp_nondef,
            input_preprocessing=RANDOM_CRISP_NONDEF,
            output_preprocessing=CANONICAL_CRISP_NONDEF,
        ),
        ExperimentSpec(
            dataset="zinc",
            task="smiles_no_stereo_to_smiles_no_stereo",
            config=zinc_smiles_no_stereo_to_smiles_no_stereo,
            input_preprocessing=STRIPPED_RANDOM_SMILES,
            output_preprocessing=STRIPPED_CANONICAL_SMILES,
        ),
        ExperimentSpec(
            dataset="zinc",
            task="crisp_def_to_smiles",
            config=zinc_crisp_def_to_smiles,
            input_preprocessing=RANDOM_CRISP_DEF,
            output_preprocessing=CANONICAL_SMILES,
        ),
        ExperimentSpec(
            dataset="zinc",
            task="smiles_to_crisp_def",
            config=zinc_smiles_to_crisp_def,
            input_preprocessing=RANDOM_SMILES,
            output_preprocessing=CANONICAL_CRISP_DEF,
        ),
        # ── USPTO forward reaction prediction experiments (6 conditions) ──
        ExperimentSpec(
            dataset="USPTO_STEREO",
            task="smiles_to_smiles",
            config=uspto_smiles_to_smiles,
            input_preprocessing=RANDOM_SMILES,
            output_preprocessing=CANONICAL_SMILES,
        ),
        ExperimentSpec(
            dataset="USPTO_STEREO",
            task="crisp_def_to_crisp_def",
            config=uspto_crisp_def_to_crisp_def,
            input_preprocessing=RANDOM_CRISP_DEF,
            output_preprocessing=CANONICAL_CRISP_DEF,
        ),
        ExperimentSpec(
            dataset="USPTO_STEREO",
            task="crisp_nondef_to_crisp_nondef",
            config=uspto_crisp_nondef_to_crisp_nondef,
            input_preprocessing=RANDOM_CRISP_NONDEF,
            output_preprocessing=CANONICAL_CRISP_NONDEF,
        ),
        ExperimentSpec(
            dataset="USPTO_STEREO",
            task="smiles_no_stereo_to_smiles_no_stereo",
            config=uspto_smiles_no_stereo_to_smiles_no_stereo,
            input_preprocessing=STRIPPED_RANDOM_SMILES,
            output_preprocessing=STRIPPED_CANONICAL_SMILES,
        ),
        ExperimentSpec(
            dataset="USPTO_STEREO",
            task="crisp_def_to_smiles",
            config=uspto_crisp_def_to_smiles,
            input_preprocessing=RANDOM_CRISP_DEF,
            output_preprocessing=CANONICAL_SMILES,
        ),
        ExperimentSpec(
            dataset="USPTO_STEREO",
            task="smiles_to_crisp_def",
            config=uspto_smiles_to_crisp_def,
            input_preprocessing=RANDOM_SMILES,
            output_preprocessing=CANONICAL_CRISP_DEF,
        ),
    ]


def run_experiment(
    experiment: ExperimentSpec,
    train_data: list[str],
    val_data: list[str],
    *,
    test_data: list[str] | None = None,
) -> dict[str, Any]:
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

    # Compute top tokens from a sample of training data for per-token accuracy tracking
    sample_size = min(10000, len(train_data))
    sample_pairs = generate_training_data(
        train_data[:sample_size],
        mode=mode,
        input_smiles_preprocessing=input_preprocessing,
        output_smiles_preprocessing=output_preprocessing,
        verbose=False,
    )
    top_tokens = compute_top_tokens(sample_pairs, tokenizer, top_k=30)

    trainer = SmilesTrainer(
        config=config,
        tokenizer=tokenizer,
        train_data=train_data,
        mode=mode,
        val_data=val_data,
        test_data=processed_test,
        input_smiles_preprocessing=input_preprocessing,
        output_smiles_preprocessing=output_preprocessing,
        top_tokens=top_tokens,
    )

    trainer.train()

    # Evaluate on test set if available
    results: dict[str, Any] = {}
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
            f.write(
                f"non_stereo_exact_match\t{results.get('non_stereo_exact_match', '')}\n"
            )
            f.write(f"stereo_accuracy\t{results.get('stereo_accuracy', '')}\n")

    return results


def run_experiment_multi_seed(
    experiment: ExperimentSpec,
    train_data: list[str],
    val_data: list[str],
    *,
    test_data: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Run a single experiment across multiple random seeds.

    Each seed run gets its own subdirectory under the experiment's output_dir
    (e.g. ``results/<task>/seed_42/``).  After all seeds complete, results are
    aggregated into a summary file.

    Args:
        experiment (ExperimentSpec): Experiment specification.
        train_data (List[str]): Raw training strings.
        val_data (List[str]): Raw validation strings.
        test_data (Optional[List[str]]): Raw test strings.
        seeds (Optional[List[int]]): Seeds to run.  Defaults to [42, 123, 456].

    Returns:
        List[Dict[str, Any]]: List of per-seed result dicts.
    """
    if seeds is None:
        seeds = [42, 123, 456]

    all_results: list[dict[str, Any]] = []
    base_output_dir = experiment.config.output_dir

    for seed in seeds:
        print(f"\n{'=' * 80}")
        print(f"  SEED {seed} — {experiment.dataset} / {experiment.task}")
        print(f"{'=' * 80}\n")

        # Create a seed-specific config with adjusted output dirs
        seed_output_dir = str(Path(base_output_dir) / f"seed_{seed}")
        seed_config = CustomSmallConfig(
            **{
                **experiment.config.__dict__,
                "output_dir": seed_output_dir,
                "logging_dir": str(Path(seed_output_dir) / "logs"),
                "analysis_dir": str(Path(seed_output_dir) / "analysis"),
                "seed": seed,
            }
        )

        seed_experiment = ExperimentSpec(
            dataset=experiment.dataset,
            task=experiment.task,
            config=seed_config,
            input_preprocessing=experiment.input_preprocessing,
            output_preprocessing=experiment.output_preprocessing,
        )

        results = run_experiment(
            seed_experiment,
            train_data,
            val_data,
            test_data=test_data,
        )
        results["seed"] = seed
        all_results.append(results)

    return all_results


def aggregate_results(
    all_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Aggregate multi-seed results into a summary TSV file.

    Args:
        all_results (List[Dict[str, Any]]): List of per-seed result dicts.
        output_path (Path): Path to write the summary TSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(
            "seed\texact_match\ttoken_accuracy\tnon_stereo_exact_match\tstereo_accuracy\n"
        )
        for result in all_results:
            f.write(
                f"{result.get('seed', '')}\t"
                f"{result.get('exact_match', '')}\t"
                f"{result.get('token_accuracy', '')}\t"
                f"{result.get('non_stereo_exact_match', '')}\t"
                f"{result.get('stereo_accuracy', '')}\n"
            )
    print(f"✓ Aggregated results saved to {output_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
        help="Directory containing Jin_USPTO_1product_{train,valid,test}.txt files.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=245000,
        help="Number of ZINC molecules to use for training (rest go to validation).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds for multi-seed runs.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiments = get_default_experiments()

    # ── Load ZINC data (SMILES strings, split into train/val) ─────────────
    print("Loading ZINC dataset...")
    zinc_smiles = _read_lines(args.zinc_smiles_path)
    zinc_train = zinc_smiles[: args.train_size]
    zinc_val = zinc_smiles[args.train_size :]
    print(f"  ZINC train: {len(zinc_train)}  |  val: {len(zinc_val)}")

    # ── Load USPTO data (pre-split reaction SMILES) ──────────────────────
    uspto_train_path = args.uspto_dir / "Jin_USPTO_1product_train.txt"
    uspto_val_path = args.uspto_dir / "Jin_USPTO_1product_valid.txt"
    uspto_test_path = args.uspto_dir / "Jin_USPTO_1product_test.txt"

    print("Loading USPTO_STEREO dataset...")
    uspto_train = _read_lines(uspto_train_path)
    uspto_val = _read_lines(uspto_val_path)
    uspto_test = _read_lines(uspto_test_path)
    print(
        f"  USPTO train: {len(uspto_train)}  |  val: {len(uspto_val)}"
        f"  |  test: {len(uspto_test)}"
    )

    # ── Run experiments with multi-seed ─────────────────────────────────
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT {i}/{len(experiments)}: {exp.dataset} / {exp.task}")
        print(f"{'=' * 80}\n")

        if exp.dataset == "zinc":
            all_results = run_experiment_multi_seed(
                exp, zinc_train, zinc_val, seeds=args.seeds
            )
        elif exp.dataset == "USPTO_STEREO":
            all_results = run_experiment_multi_seed(
                exp,
                uspto_train,
                uspto_val,
                test_data=uspto_test,
                seeds=args.seeds,
            )
        else:
            raise ValueError(f"Unknown dataset '{exp.dataset}'")

        # Save aggregated results
        summary_path = Path(exp.config.output_dir) / "seed_summary.txt"
        aggregate_results(all_results, summary_path)

    print(f"\n{'=' * 80}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
