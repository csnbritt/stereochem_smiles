import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.training.trainer import CustomSmallConfig

# ── Common training parameters ────────────────────────────────────────────────
# Fixed-step training (no early stopping) for consistent accuracy-over-time curves.
_ZINC_MAX_STEPS = 50000
_USPTO_MAX_STEPS = 50000
_EVAL_STEPS = 5000
_SAVE_STEPS = 5000

# Default fraction of each dataset used for training (remainder goes to
# validation). Referenced as train_frac=_TRAIN_FRAC in each config below;
# override per experiment by passing a different literal.
_TRAIN_FRAC = 0.95

# ── ZINC translation experiments (8 conditions) ───────────────────────────────

zinc_smiles_to_smiles = CustomSmallConfig(
    output_dir="results/zinc_smiles_to_smiles/",
    logging_dir="results/zinc_smiles_to_smiles/logs/",
    analysis_dir="results/zinc_smiles_to_smiles/analysis/",
    max_steps=_ZINC_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
    train_frac=_TRAIN_FRAC,
)

zinc_crisp_def_to_crisp_def = CustomSmallConfig(
    output_dir="results/zinc_crisp_def_to_crisp_def/",
    logging_dir="results/zinc_crisp_def_to_crisp_def/logs/",
    analysis_dir="results/zinc_crisp_def_to_crisp_def/analysis/",
    max_steps=_ZINC_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
    train_frac=_TRAIN_FRAC,
)

zinc_crisp_nondef_to_crisp_nondef = CustomSmallConfig(
    output_dir="results/zinc_crisp_nondef_to_crisp_nondef/",
    logging_dir="results/zinc_crisp_nondef_to_crisp_nondef/logs/",
    analysis_dir="results/zinc_crisp_nondef_to_crisp_nondef/analysis/",
    max_steps=_ZINC_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
    train_frac=_TRAIN_FRAC,
)

zinc_smiles_no_stereo_to_smiles_no_stereo = CustomSmallConfig(
    output_dir="results/zinc_smiles_no_stereo_to_smiles_no_stereo/",
    logging_dir="results/zinc_smiles_no_stereo_to_smiles_no_stereo/logs/",
    analysis_dir="results/zinc_smiles_no_stereo_to_smiles_no_stereo/analysis/",
    max_steps=_ZINC_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
    train_frac=_TRAIN_FRAC,
)

zinc_crisp_def_to_smiles = CustomSmallConfig(
    output_dir="results/zinc_crisp_def_to_smiles/",
    logging_dir="results/zinc_crisp_def_to_smiles/logs/",
    analysis_dir="results/zinc_crisp_def_to_smiles/analysis/",
    max_steps=_ZINC_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
    train_frac=_TRAIN_FRAC,
)

zinc_smiles_to_crisp_def = CustomSmallConfig(
    output_dir="results/zinc_smiles_to_crisp_def/",
    logging_dir="results/zinc_smiles_to_crisp_def/logs/",
    analysis_dir="results/zinc_smiles_to_crisp_def/analysis/",
    max_steps=_ZINC_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
    train_frac=_TRAIN_FRAC,
)

zinc_crisp_nondef_to_smiles = CustomSmallConfig(
    output_dir="results/zinc_crisp_nondef_to_smiles/",
    logging_dir="results/zinc_crisp_nondef_to_smiles/logs/",
    analysis_dir="results/zinc_crisp_nondef_to_smiles/analysis/",
    max_steps=_ZINC_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
)

zinc_smiles_to_crisp_nondef = CustomSmallConfig(
    output_dir="results/zinc_smiles_to_crisp_nondef/",
    logging_dir="results/zinc_smiles_to_crisp_nondef/logs/",
    analysis_dir="results/zinc_smiles_to_crisp_nondef/analysis/",
    max_steps=_ZINC_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
)

# ── USPTO forward reaction prediction experiments (8 conditions) ──────────────

uspto_smiles_to_smiles = CustomSmallConfig(
    output_dir="results/uspto_smiles_to_smiles/",
    logging_dir="results/uspto_smiles_to_smiles/logs/",
    analysis_dir="results/uspto_smiles_to_smiles/analysis/",
    max_steps=_USPTO_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
    train_frac=_TRAIN_FRAC,
)

uspto_crisp_def_to_crisp_def = CustomSmallConfig(
    output_dir="results/uspto_crisp_def_to_crisp_def/",
    logging_dir="results/uspto_crisp_def_to_crisp_def/logs/",
    analysis_dir="results/uspto_crisp_def_to_crisp_def/analysis/",
    max_steps=_USPTO_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
    train_frac=_TRAIN_FRAC,
)

uspto_crisp_nondef_to_crisp_nondef = CustomSmallConfig(
    output_dir="results/uspto_crisp_nondef_to_crisp_nondef/",
    logging_dir="results/uspto_crisp_nondef_to_crisp_nondef/logs/",
    analysis_dir="results/uspto_crisp_nondef_to_crisp_nondef/analysis/",
    max_steps=_USPTO_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
    train_frac=_TRAIN_FRAC,
)

uspto_smiles_no_stereo_to_smiles_no_stereo = CustomSmallConfig(
    output_dir="results/uspto_smiles_no_stereo_to_smiles_no_stereo/",
    logging_dir="results/uspto_smiles_no_stereo_to_smiles_no_stereo/logs/",
    analysis_dir="results/uspto_smiles_no_stereo_to_smiles_no_stereo/analysis/",
    max_steps=_USPTO_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
    train_frac=_TRAIN_FRAC,
)

uspto_crisp_def_to_smiles = CustomSmallConfig(
    output_dir="results/uspto_crisp_def_to_smiles/",
    logging_dir="results/uspto_crisp_def_to_smiles/logs/",
    analysis_dir="results/uspto_crisp_def_to_smiles/analysis/",
    max_steps=_USPTO_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
    train_frac=_TRAIN_FRAC,
)

uspto_smiles_to_crisp_def = CustomSmallConfig(
    output_dir="results/uspto_smiles_to_crisp_def/",
    logging_dir="results/uspto_smiles_to_crisp_def/logs/",
    analysis_dir="results/uspto_smiles_to_crisp_def/analysis/",
    max_steps=_USPTO_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
    train_frac=_TRAIN_FRAC,
)

uspto_crisp_nondef_to_smiles = CustomSmallConfig(
    output_dir="results/uspto_crisp_nondef_to_smiles/",
    logging_dir="results/uspto_crisp_nondef_to_smiles/logs/",
    analysis_dir="results/uspto_crisp_nondef_to_smiles/analysis/",
    max_steps=_USPTO_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
)

uspto_smiles_to_crisp_nondef = CustomSmallConfig(
    output_dir="results/uspto_smiles_to_crisp_nondef/",
    logging_dir="results/uspto_smiles_to_crisp_nondef/logs/",
    analysis_dir="results/uspto_smiles_to_crisp_nondef/analysis/",
    max_steps=_USPTO_MAX_STEPS,
    use_early_stopping=False,
    eval_steps=_EVAL_STEPS,
    save_steps=_SAVE_STEPS,
)