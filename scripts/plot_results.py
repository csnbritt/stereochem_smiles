#!/usr/bin/env python3
"""
Generate publication-quality figures from CRISP SMILES experiment results.

Reads the results/ directory produced by generate_manuscript_data.py and
produces PDF/SVG figures suitable for manuscript inclusion.

Usage:
    python scripts/plot_results.py --results-dir results/ --output-dir figures/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Publication style ────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
})

# Colorblind-friendly palette
PALETTE = sns.color_palette("colorblind", 8)

# ── Experiment metadata ──────────────────────────────────────────────────────

EXPERIMENT_LABELS: Dict[str, str] = {
    "zinc_smiles_to_smiles": "SMILES→SMILES",
    "zinc_crisp_def_to_crisp_def": "CRISP(def)→CRISP(def)",
    "zinc_crisp_nondef_to_crisp_nondef": "CRISP(nondef)→CRISP(nondef)",
    "zinc_smiles_no_stereo_to_smiles_no_stereo": "SMILES(no stereo)→SMILES(no stereo)",
    "zinc_crisp_def_to_smiles": "CRISP(def)→SMILES",
    "zinc_smiles_to_crisp_def": "SMILES→CRISP(def)",
    "uspto_smiles_to_smiles": "SMILES→SMILES",
    "uspto_crisp_def_to_crisp_def": "CRISP(def)→CRISP(def)",
    "uspto_crisp_nondef_to_crisp_nondef": "CRISP(nondef)→CRISP(nondef)",
    "uspto_smiles_no_stereo_to_smiles_no_stereo": "SMILES(no stereo)→SMILES(no stereo)",
    "uspto_crisp_def_to_smiles": "CRISP(def)→SMILES",
    "uspto_smiles_to_crisp_def": "SMILES→CRISP(def)",
}

# Order for consistent plotting
ZINC_ORDER = [
    "zinc_smiles_to_smiles",
    "zinc_crisp_def_to_crisp_def",
    "zinc_crisp_nondef_to_crisp_nondef",
    "zinc_smiles_no_stereo_to_smiles_no_stereo",
    "zinc_crisp_def_to_smiles",
    "zinc_smiles_to_crisp_def",
]
USPTO_ORDER = [
    "uspto_smiles_to_smiles",
    "uspto_crisp_def_to_crisp_def",
    "uspto_crisp_nondef_to_crisp_nondef",
    "uspto_smiles_no_stereo_to_smiles_no_stereo",
    "uspto_crisp_def_to_smiles",
    "uspto_smiles_to_crisp_def",
]

METRIC_COLUMNS = [
    "step",
    "exact_match",
    "token_accuracy",
    "stereo_token_accuracy",
    "non_stereo_exact_match",
    "stereo_accuracy",
]


# ── Data loading ─────────────────────────────────────────────────────────────


def _load_eval_metrics(analysis_dir: Path) -> Optional[pd.DataFrame]:
    """Load eval_metrics_by_step.txt from a seed's analysis directory."""
    path = analysis_dir / "eval_metrics_by_step.txt"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t", header=None, names=METRIC_COLUMNS)
        return df
    except Exception:
        return None


def _load_eval_loss(analysis_dir: Path) -> Optional[pd.DataFrame]:
    """Load eval_loss_by_step.txt from a seed's analysis directory."""
    path = analysis_dir / "eval_loss_by_step.txt"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t", header=None, names=["step", "eval_loss"])
        return df
    except Exception:
        return None


def _load_train_loss(analysis_dir: Path) -> Optional[pd.DataFrame]:
    """Load train_loss_by_step.txt from a seed's analysis directory."""
    path = analysis_dir / "train_loss_by_step.txt"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t", header=None, names=["step", "train_loss"])
        return df
    except Exception:
        return None


def _load_per_token_accuracy(analysis_dir: Path, step: int) -> Optional[pd.DataFrame]:
    """Load per_token_accuracy_step_{step}.txt."""
    path = analysis_dir / f"per_token_accuracy_step_{step}.txt"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(
            path, sep="\t", header=None,
            names=["token", "correct", "total", "accuracy"],
        )
        return df
    except Exception:
        return None


def load_all_results(results_dir: Path) -> Dict[str, Dict[int, dict]]:
    """
    Load all experiment results from the results directory.

    Returns:
        Dict mapping experiment_name → seed → {metrics, eval_loss, train_loss}.
    """
    results: Dict[str, Dict[int, dict]] = {}
    if not results_dir.exists():
        return results

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name
        results[exp_name] = {}

        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = int(seed_dir.name.split("_")[1])
            analysis_dir = seed_dir / "analysis"
            if not analysis_dir.exists():
                continue

            results[exp_name][seed] = {
                "metrics": _load_eval_metrics(analysis_dir),
                "eval_loss": _load_eval_loss(analysis_dir),
                "train_loss": _load_train_loss(analysis_dir),
                "analysis_dir": analysis_dir,
            }

    return results


def aggregate_metric(
    results: Dict[str, Dict[int, dict]],
    experiment: str,
    metric: str,
) -> Optional[pd.DataFrame]:
    """
    Aggregate a metric across seeds for a given experiment.

    Returns:
        DataFrame with columns: step, mean, std, n_seeds.
    """
    if experiment not in results:
        return None

    dfs = []
    for data in results[experiment].values():
        df = data.get("metrics")
        if df is not None and metric in df.columns:
            dfs.append(df[["step", metric]].rename(columns={metric: "value"}))

    if not dfs:
        return None

    combined = pd.concat(dfs)
    agg = combined.groupby("step")["value"].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["step", "mean", "std", "n_seeds"]
    return agg


# ── Figure 1: Learning curves ────────────────────────────────────────────────


def plot_learning_curves(
    results: Dict[str, Dict[int, dict]],
    output_dir: Path,
    dataset: str = "zinc",
) -> None:
    """Plot exact match and stereo token accuracy vs training step."""
    order = ZINC_ORDER if dataset == "zinc" else USPTO_ORDER
    experiments = [e for e in order if e in results]
    if not experiments:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for i, exp in enumerate(experiments):
        label = EXPERIMENT_LABELS.get(exp, exp)
        color = PALETTE[i % len(PALETTE)]

        # Exact match
        agg = aggregate_metric(results, exp, "exact_match")
        if agg is not None:
            axes[0].plot(agg["step"], agg["mean"], label=label, color=color)
            if agg["n_seeds"].max() > 1:
                axes[0].fill_between(
                    agg["step"],
                    agg["mean"] - agg["std"],
                    agg["mean"] + agg["std"],
                    alpha=0.15, color=color,
                )

        # Stereo token accuracy
        agg = aggregate_metric(results, exp, "stereo_token_accuracy")
        if agg is not None:
            axes[1].plot(agg["step"], agg["mean"], label=label, color=color)
            if agg["n_seeds"].max() > 1:
                axes[1].fill_between(
                    agg["step"],
                    agg["mean"] - agg["std"],
                    agg["mean"] + agg["std"],
                    alpha=0.15, color=color,
                )

    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Exact Match")
    axes[0].set_title("Exact Match Accuracy")
    axes[0].legend(fontsize=6, loc="best", framealpha=0.9)
    axes[0].set_ylim(0, 1.05)

    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Stereo Token Accuracy")
    axes[1].set_title("Stereo Token Accuracy")
    axes[1].legend(fontsize=6, loc="best", framealpha=0.9)
    axes[1].set_ylim(0, 1.05)

    dataset_label = "ZINC-250K" if dataset == "zinc" else "USPTO"
    fig.suptitle(f"{dataset_label} — Learning Curves", fontsize=11, y=1.02)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"learning_curves_{dataset}.{ext}")
    plt.close(fig)
    print(f"  ✓ learning_curves_{dataset}.pdf")


# ── Figure 2: Final performance bar chart ────────────────────────────────────


def plot_final_performance(
    results: Dict[str, Dict[int, dict]],
    output_dir: Path,
    dataset: str = "zinc",
) -> None:
    """Grouped bar chart of final metrics for each experiment."""
    order = ZINC_ORDER if dataset == "zinc" else USPTO_ORDER
    experiments = [e for e in order if e in results]
    if not experiments:
        return

    metrics_to_plot = ["exact_match", "stereo_token_accuracy", "non_stereo_exact_match", "stereo_accuracy"]
    metric_labels = ["Exact Match", "Stereo Token Acc", "Non-Stereo Exact", "Stereo Accuracy"]

    # Collect final-step values per experiment per metric
    data = []
    for exp in experiments:
        label = EXPERIMENT_LABELS.get(exp, exp)
        for metric, mlabel in zip(metrics_to_plot, metric_labels):
            agg = aggregate_metric(results, exp, metric)
            if agg is not None and len(agg) > 0:
                final = agg.iloc[-1]
                data.append({
                    "experiment": label,
                    "metric": mlabel,
                    "mean": final["mean"],
                    "std": final["std"] if not np.isnan(final["std"]) else 0.0,
                })

    if not data:
        return

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(experiments))
    width = 0.2

    for i, (metric, mlabel) in enumerate(zip(metrics_to_plot, metric_labels)):
        subset = df[df["metric"] == mlabel]
        means = [subset[subset["experiment"] == EXPERIMENT_LABELS.get(e, e)]["mean"].values[0]
                 if len(subset[subset["experiment"] == EXPERIMENT_LABELS.get(e, e)]) > 0 else 0
                 for e in experiments]
        stds = [subset[subset["experiment"] == EXPERIMENT_LABELS.get(e, e)]["std"].values[0]
                if len(subset[subset["experiment"] == EXPERIMENT_LABELS.get(e, e)]) > 0 else 0
                for e in experiments]
        ax.bar(x + i * width, means, width, yerr=stds, label=mlabel,
               color=PALETTE[i], capsize=3, error_kw={"linewidth": 0.8})

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([EXPERIMENT_LABELS.get(e, e) for e in experiments],
                       rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"{'ZINC-250K' if dataset == 'zinc' else 'USPTO'} — Final Performance")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"final_performance_{dataset}.{ext}")
    plt.close(fig)
    print(f"  ✓ final_performance_{dataset}.pdf")


# ── Figure 3: Per-token accuracy ─────────────────────────────────────────────


def plot_per_token_accuracy(
    results: Dict[str, Dict[int, dict]],
    output_dir: Path,
    dataset: str = "zinc",
    top_n: int = 40,
) -> None:
    """Bar chart of per-token accuracy at the final eval step, highlighting stereo tokens."""
    order = ZINC_ORDER if dataset == "zinc" else USPTO_ORDER
    experiments = [e for e in order if e in results]
    if not experiments:
        return

    # Find the latest step with per-token data across all experiments
    all_token_data: Dict[str, pd.DataFrame] = {}
    for exp in experiments:
        for data in results[exp].values():
            analysis_dir = data.get("analysis_dir")
            if analysis_dir is None:
                continue
            # Find latest per_token_accuracy file
            pta_files = sorted(analysis_dir.glob("per_token_accuracy_step_*.txt"))
            if not pta_files:
                continue
            latest = pta_files[-1]
            step = int(latest.stem.split("_")[-1])
            df = _load_per_token_accuracy(analysis_dir, step)
            if df is not None:
                if exp not in all_token_data:
                    all_token_data[exp] = df
                else:
                    # Average across seeds
                    all_token_data[exp] = pd.concat([all_token_data[exp], df]).groupby("token").agg(
                        {"correct": "sum", "total": "sum"}
                    ).reset_index()
                    all_token_data[exp]["accuracy"] = (
                        all_token_data[exp]["correct"] / all_token_data[exp]["total"]
                    )

    if not all_token_data:
        return

    # Get top-N tokens by total frequency across all experiments
    all_totals: Dict[str, int] = {}
    for df in all_token_data.values():
        for _, row in df.iterrows():
            all_totals[row["token"]] = all_totals.get(row["token"], 0) + row["total"]
    top_tokens = sorted(all_totals, key=lambda t: -all_totals[t])[:top_n]

    # Build comparison DataFrame
    plot_data = []
    for exp in experiments:
        label = EXPERIMENT_LABELS.get(exp, exp)
        df = all_token_data.get(exp)
        if df is None:
            continue
        for token in top_tokens:
            row = df[df["token"] == token]
            acc = row["accuracy"].values[0] if len(row) > 0 else 0.0
            is_stereo = (
                "@" in token
                or token in ("/", "\\")
                or token.startswith(("[STEREO_", "[DB_STEREO_"))
            )
            plot_data.append({
                "token": token,
                "experiment": label,
                "accuracy": acc,
                "is_stereo": is_stereo,
            })

    if not plot_data:
        return

    df_plot = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(top_tokens))
    width = 0.8 / len(experiments)

    for i, exp in enumerate(experiments):
        label = EXPERIMENT_LABELS.get(exp, exp)
        subset = df_plot[df_plot["experiment"] == label]
        accs = [subset[subset["token"] == t]["accuracy"].values[0]
                if len(subset[subset["token"] == t]) > 0 else 0
                for t in top_tokens]
        ax.bar(x + i * width, accs, width, label=label, color=PALETTE[i])

    # Highlight stereo tokens on x-axis
    for j, token in enumerate(top_tokens):
        is_stereo = (
            "@" in token
            or token in ("/", "\\")
            or token.startswith(("[STEREO_", "[DB_STEREO_"))
        )
        if is_stereo:
            ax.axvspan(j - 0.4, j + 0.4, alpha=0.08, color="red", zorder=0)

    ax.set_xticks(x + width * (len(experiments) - 1) / 2)
    ax.set_xticklabels(top_tokens, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title(
        f"{'ZINC-250K' if dataset == 'zinc' else 'USPTO'} — Per-Token Accuracy "
        f"(shaded = stereo tokens)"
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"per_token_accuracy_{dataset}.{ext}")
    plt.close(fig)
    print(f"  ✓ per_token_accuracy_{dataset}.pdf")


# ── Figure 4: Loss curves ────────────────────────────────────────────────────


def plot_loss_curves(
    results: Dict[str, Dict[int, dict]],
    output_dir: Path,
    dataset: str = "zinc",
) -> None:
    """Plot training and eval loss vs training step."""
    order = ZINC_ORDER if dataset == "zinc" else USPTO_ORDER
    experiments = [e for e in order if e in results]
    if not experiments:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for i, exp in enumerate(experiments):
        label = EXPERIMENT_LABELS.get(exp, exp)
        color = PALETTE[i % len(PALETTE)]

        # Train loss
        train_dfs = []
        for data in results[exp].values():
            df = data.get("train_loss")
            if df is not None:
                train_dfs.append(df)
        if train_dfs:
            combined = pd.concat(train_dfs)
            agg = combined.groupby("step")["train_loss"].agg(["mean", "std"]).reset_index()
            axes[0].plot(agg["step"], agg["mean"], label=label, color=color)
            if len(train_dfs) > 1:
                axes[0].fill_between(
                    agg["step"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                    alpha=0.15, color=color,
                )

        # Eval loss
        eval_dfs = []
        for data in results[exp].values():
            df = data.get("eval_loss")
            if df is not None:
                eval_dfs.append(df)
        if eval_dfs:
            combined = pd.concat(eval_dfs)
            agg = combined.groupby("step")["eval_loss"].agg(["mean", "std"]).reset_index()
            axes[1].plot(agg["step"], agg["mean"], label=label, color=color)
            if len(eval_dfs) > 1:
                axes[1].fill_between(
                    agg["step"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                    alpha=0.15, color=color,
                )

    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend(fontsize=6, loc="best", framealpha=0.9)

    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation Loss")
    axes[1].legend(fontsize=6, loc="best", framealpha=0.9)

    dataset_label = "ZINC-250K" if dataset == "zinc" else "USPTO"
    fig.suptitle(f"{dataset_label} — Loss Curves", fontsize=11, y=1.02)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"loss_curves_{dataset}.{ext}")
    plt.close(fig)
    print(f"  ✓ loss_curves_{dataset}.pdf")


# ── Figure 5: Stereo vs non-stereo scatter ───────────────────────────────────


def plot_stereo_scatter(
    results: Dict[str, Dict[int, dict]],
    output_dir: Path,
) -> None:
    """Scatter plot of non-stereo exact match vs stereo accuracy."""
    all_experiments = list(results.keys())
    if not all_experiments:
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for i, exp in enumerate(all_experiments):
        label = EXPERIMENT_LABELS.get(exp, exp)
        color = PALETTE[i % len(PALETTE)]
        dataset = "ZINC" if exp.startswith("zinc") else "USPTO"
        marker = "o" if dataset == "ZINC" else "s"

        for seed, data in results[exp].items():
            df = data.get("metrics")
            if df is None or len(df) == 0:
                continue
            final = df.iloc[-1]
            ax.scatter(
                final["non_stereo_exact_match"],
                final["stereo_accuracy"],
                color=color, marker=marker, s=40, alpha=0.8,
                edgecolors="black", linewidths=0.5,
                label=f"{label} ({dataset})" if seed == next(iter(results[exp].keys())) else "",
            )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8, label="y = x")

    ax.set_xlabel("Non-Stereo Exact Match")
    ax.set_ylabel("Stereo Accuracy")
    ax.set_title("Connectivity vs Stereochemistry")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=5, loc="lower right", framealpha=0.9, ncol=2)
    ax.set_aspect("equal")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"stereo_scatter.{ext}")
    plt.close(fig)
    print("  ✓ stereo_scatter.pdf")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures from experiment results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Path to the results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to save figures.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.results_dir}...")
    results = load_all_results(args.results_dir)

    if not results:
        print("No results found. Check --results-dir path.")
        return 1

    print(f"Found {len(results)} experiments:")
    for exp, seeds in results.items():
        print(f"  {exp}: {len(seeds)} seed(s)")

    print(f"\nGenerating figures in {args.output_dir}/...")

    for dataset in ("zinc", "uspto"):
        plot_learning_curves(results, args.output_dir, dataset)
        plot_final_performance(results, args.output_dir, dataset)
        plot_per_token_accuracy(results, args.output_dir, dataset)
        plot_loss_curves(results, args.output_dir, dataset)

    plot_stereo_scatter(results, args.output_dir)

    print(f"\nDone. Figures saved to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
