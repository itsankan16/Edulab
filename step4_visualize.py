# -*- coding: utf-8 -*-
"""
step4_visualize.py - EduBench-Local Pipeline Step 4
=====================================================
Reads evaluation data from results/ and generates a comprehensive, publication-grade
suite of exactly 18 unique visualizations with zero duplication and strict naming conventions:

1. Qwen 2.5 (3B) [Exactly 6 Files]:
   - qwen_bars.png         : 3-panel horizontal bar chart with error bars
   - qwen_radar.png        : Multi-metric polar radar profile across 5 datasets
   - qwen_boxplots.png     : Score distributions boxplots by dataset
   - qwen_heatmap.png      : Metric score heatmap (Dataset x Metric)
   - qwen_latency.png      : Generation latency by dataset (mean +- std dev)
   - qwen_correlations.png : Metric correlation analysis scatter & regressions

2. Google Gemini [Exactly 6 Files]:
   - gemini_bars.png       : 3-panel horizontal bar chart with error bars
   - gemini_radar.png      : Multi-metric polar radar profile across 5 datasets
   - gemini_boxplots.png   : Score distributions boxplots by dataset
   - gemini_heatmap.png    : Metric score heatmap (Dataset x Metric)
   - gemini_latency.png    : Generation latency by dataset (mean +- std dev)
   - gemini_correlations.png : Metric correlation analysis scatter & regressions

3. Head-to-Head Comparison [Exactly 6 Files]:
   - comparison_bars.png      : Grouped horizontal bar charts comparing Qwen vs Gemini
   - comparison_scorecard.png : Executive comparative scorecard and dataset win matrix
   - comparison_f1.png        : Head-to-head dataset F1 comparison with winner badges
   - comparison_latency.png   : Generation latency dual-panel comparison
   - comparison_radar.png     : Dual-model comparative radar footprint
   - comparison_heatmap.png   : Side-by-side comparative heatmap (Qwen vs Gemini)

Total: Exactly 18 files. No generic or legacy filenames are created or retained.

Usage:
  python step4_visualize.py [--dpi 300]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import config
from generate_clean_white_visualizations import (
    apply_clean_white_style,
    load_and_preprocess_data,
    plot_clean_white_qwen_performance_bars,
    plot_clean_white_multi_metric_radar,
    plot_clean_white_score_distributions,
    plot_clean_white_score_heatmap,
    plot_clean_white_generation_latency,
    plot_clean_white_metric_correlations,
    plot_clean_white_gemini_performance_bars,
    plot_clean_white_gemini_radar,
    plot_clean_white_gemini_score_distributions,
    plot_clean_white_gemini_metric_correlations,
    plot_clean_white_model_comparison_bars,
    plot_clean_white_executive_scorecard,
    plot_clean_white_comparison_f1,
    plot_clean_white_generation_latency_comparison,
    plot_clean_white_comparison_radar,
    plot_clean_white_score_heatmap_comparison,
)

# ---------------------------------------------------------------------------
# Strict Canonical Filename Constants (Exactly 18 Files)
# ---------------------------------------------------------------------------
QWEN_FIGURE_NAMES: List[str] = [
    "qwen_bars.png",
    "qwen_radar.png",
    "qwen_boxplots.png",
    "qwen_heatmap.png",
    "qwen_latency.png",
    "qwen_correlations.png",
]

GEMINI_FIGURE_NAMES: List[str] = [
    "gemini_bars.png",
    "gemini_radar.png",
    "gemini_boxplots.png",
    "gemini_heatmap.png",
    "gemini_latency.png",
    "gemini_correlations.png",
]

COMPARISON_FIGURE_NAMES: List[str] = [
    "comparison_bars.png",
    "comparison_scorecard.png",
    "comparison_f1.png",
    "comparison_latency.png",
    "comparison_radar.png",
    "comparison_heatmap.png",
]

ALL_18_FIGURE_NAMES: List[str] = QWEN_FIGURE_NAMES + GEMINI_FIGURE_NAMES + COMPARISON_FIGURE_NAMES
assert len(ALL_18_FIGURE_NAMES) == 18, f"Expected exactly 18 figure names, got {len(ALL_18_FIGURE_NAMES)}"
assert len(set(ALL_18_FIGURE_NAMES)) == 18, "Duplicate filenames detected in canonical list"

FIGURES_DIR: Path = config.FIGURES_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Generation Routines
# ---------------------------------------------------------------------------
def generate_qwen_figures(qwen_df, gemini_df, dpi: int = 300) -> List[Path]:
    """Generates strictly the 6 Qwen individual visualization files."""
    paths = []
    print("\n--- Generating Qwen 2.5 (3B) Visualizations (6 Files) ---")

    print("[1/6] qwen_bars.png")
    paths.append(plot_clean_white_qwen_performance_bars(qwen_df, "qwen_bars.png"))

    print("[2/6] qwen_radar.png")
    paths.append(plot_clean_white_multi_metric_radar(qwen_df, gemini_df, "qwen_radar.png"))

    print("[3/6] qwen_boxplots.png")
    paths.append(plot_clean_white_score_distributions(qwen_df, "qwen_boxplots.png"))

    print("[4/6] qwen_heatmap.png")
    qwen_metrics = {
        "ROUGE-L": "rouge_l",
        "BERTScore F1": "bert_score_f1",
        "Judge Score (1-5)": "llm_score_1_5",
    }
    paths.append(plot_clean_white_score_heatmap(qwen_df, "Qwen 2.5 (3B)", qwen_metrics, "qwen_heatmap.png"))

    print("[5/6] qwen_latency.png")
    paths.append(plot_clean_white_generation_latency(qwen_df, "Qwen 2.5 (3B)", "qwen_latency.png", dpi=dpi))

    print("[6/6] qwen_correlations.png")
    paths.append(plot_clean_white_metric_correlations(qwen_df, "qwen_correlations.png"))

    return paths


def generate_gemini_figures(gemini_df, dpi: int = 300) -> List[Path]:
    """Generates strictly the 6 Gemini individual visualization files."""
    paths = []
    print("\n--- Generating Google Gemini Visualizations (6 Files) ---")

    print("[1/6] gemini_bars.png")
    paths.append(plot_clean_white_gemini_performance_bars(gemini_df, "gemini_bars.png"))

    print("[2/6] gemini_radar.png")
    paths.append(plot_clean_white_gemini_radar(gemini_df, "gemini_radar.png"))

    print("[3/6] gemini_boxplots.png")
    paths.append(plot_clean_white_gemini_score_distributions(gemini_df, "gemini_boxplots.png"))

    print("[4/6] gemini_heatmap.png")
    gemini_metrics = {
        "ROUGE-L": "rouge_l",
        "Token F1": "token_f1",
        "Char Similarity": "char_similarity",
        "Exact Match": "exact_match",
    }
    paths.append(plot_clean_white_score_heatmap(gemini_df, "Google Gemini", gemini_metrics, "gemini_heatmap.png"))

    print("[5/6] gemini_latency.png")
    paths.append(plot_clean_white_generation_latency(gemini_df, "Google Gemini", "gemini_latency.png", dpi=dpi))

    print("[6/6] gemini_correlations.png")
    paths.append(plot_clean_white_gemini_metric_correlations(gemini_df, "gemini_correlations.png"))

    return paths


def generate_comparison_figures(qwen_df, gemini_df, dpi: int = 300) -> List[Path]:
    """Generates strictly the 6 Head-to-Head Comparison visualization files."""
    paths = []
    print("\n--- Generating Head-to-Head Comparison Visualizations (6 Files) ---")

    print("[1/6] comparison_bars.png")
    paths.append(plot_clean_white_model_comparison_bars(qwen_df, gemini_df, "comparison_bars.png"))

    print("[2/6] comparison_scorecard.png")
    paths.append(plot_clean_white_executive_scorecard(qwen_df, gemini_df, "comparison_scorecard.png"))

    print("[3/6] comparison_f1.png")
    paths.append(plot_clean_white_comparison_f1(qwen_df, gemini_df, "comparison_f1.png", dpi=dpi))

    print("[4/6] comparison_latency.png")
    paths.append(plot_clean_white_generation_latency_comparison(qwen_df, gemini_df, "comparison_latency.png", dpi=dpi))

    print("[5/6] comparison_radar.png")
    paths.append(plot_clean_white_comparison_radar(qwen_df, gemini_df, "comparison_radar.png", dpi=dpi))

    print("[6/6] comparison_heatmap.png")
    paths.append(plot_clean_white_score_heatmap_comparison(qwen_df, gemini_df, "comparison_heatmap.png"))

    return paths


def purge_unauthorized_figures(figures_dir: Path) -> List[str]:
    """
    Deletes any PNG files in figures_dir that are not part of the canonical 18 files.
    Ensures zero duplication and no generic or legacy files linger.
    """
    removed = []
    allowed = set(ALL_18_FIGURE_NAMES)
    for p in figures_dir.glob("*.png"):
        if p.name not in allowed:
            try:
                p.unlink()
                removed.append(p.name)
            except Exception as e:
                print(f"Warning: could not delete unauthorized file {p.name}: {e}")
    if removed:
        print(f"\nPurged {len(removed)} unauthorized / legacy files: {removed}")
    return removed


def generate_all_visualizations(dpi: int = 300) -> List[Path]:
    """
    Orchestrates data loading and generates exactly 18 unique visualizations.
    Guarantees zero generic files and strictly enforces 6 Qwen, 6 Gemini, 6 Comparison.
    """
    apply_clean_white_style()
    qwen_df, gemini_df = load_and_preprocess_data()

    # Purge any non-canonical files first
    purge_unauthorized_figures(FIGURES_DIR)

    qwen_paths = generate_qwen_figures(qwen_df, gemini_df, dpi=dpi)
    gemini_paths = generate_gemini_figures(gemini_df, dpi=dpi)
    comparison_paths = generate_comparison_figures(qwen_df, gemini_df, dpi=dpi)

    all_generated = qwen_paths + gemini_paths + comparison_paths

    # Final cleanup & verification
    purge_unauthorized_figures(FIGURES_DIR)
    current_files = sorted([p.name for p in FIGURES_DIR.glob("*.png")])

    print("\n" + "=" * 68)
    print(f" EduBench Visualization Suite Generated Successfully ({len(current_files)} Files)")
    print("=" * 68)
    print(f"  • Qwen 2.5 (3B)            : {len(qwen_paths)} files (prefixed 'qwen_')")
    print(f"  • Google Gemini            : {len(gemini_paths)} files (prefixed 'gemini_')")
    print(f"  • Head-to-Head Comparison  : {len(comparison_paths)} files (prefixed 'comparison_')")
    print(f"  • Total on disk            : {len(current_files)} files (Expected: 18)")
    print("=" * 68)

    assert len(current_files) == 18, f"Expected exactly 18 files on disk, found {len(current_files)}: {current_files}"
    assert set(current_files) == set(ALL_18_FIGURE_NAMES), f"Mismatch between on-disk files and expected 18 files!"

    return all_generated


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="EduBench-Local Step 4: Generate 18 Unique Visualizations")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI resolution (default: 300)")
    args = parser.parse_args()

    print("=" * 68)
    print("EduBench-Local  |  Step 4: Visualize (Multi-Model Suite)")
    print("=" * 68)
    print(f"  Figures Directory : {FIGURES_DIR}")
    print(f"  Resolution (DPI)  : {args.dpi}")
    print()

    generate_all_visualizations(dpi=args.dpi)


if __name__ == "__main__":
    main()
