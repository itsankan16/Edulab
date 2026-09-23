"""
step4_visualize.py - EduBench-Local Pipeline Step 4
=====================================================
Reads evaluation metrics produced by step3_evaluate.py and generates a suite
of publication-quality comparative performance charts saved into figures/.

Required input files:
  results/leaderboard.csv          - per-model aggregated metrics
  results/subject_leaderboard.csv  - per-model x per-subject aggregated metrics
                                     (written by step3_evaluate.py)

Optional input file (enables 3 extra distribution charts):
  results/scored_results.csv       - per-question detail (model, subject, scores,
                                     latency, tokens)

Output figures (saved as high-resolution PNG inside figures/):
  01_overall_score_bars.png       - grouped bar chart: EM%, ROUGE-L, BERT-F1, LLM(1-5)
  02_latency_vs_accuracy.png      - scatter: avg latency vs LLM score, sized by N
  03_subject_heatmap.png          - heatmap: model x subject avg LLM(1-5) score
  04_subject_bar_comparison.png   - grouped bars per subject (LLM 1-5 score)
  05_exact_match_rate.png         - horizontal bar: exact-match rate per model
  06_subject_metric_heatmap.png   - multi-metric heatmap (ROUGE-L, BERT-F1, LLM)
  -- the following three require scored_results.csv --
  07_score_distribution.png       - violin / strip plot: LLM score distribution
  08_latency_distribution.png     - box plot: latency distribution per model
  09_tokens_vs_latency.png        - scatter: total tokens vs latency coloured by model

Usage:
  python step4_visualize.py

Dependencies:
  pip install matplotlib pandas seaborn numpy
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Guard: import third-party libs with friendly error messages
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend - no display needed
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("ERROR: matplotlib is not installed.\n  pip install matplotlib")

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy is not installed.\n  pip install numpy")

try:
    import pandas as pd
except ImportError:
    sys.exit("ERROR: pandas is not installed.\n  pip install pandas")

try:
    import seaborn as sns
except ImportError:
    sys.exit("ERROR: seaborn is not installed.\n  pip install seaborn")

import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LEADERBOARD_PATH         = config.RESULTS_DIR / "leaderboard.csv"
SUBJECT_LEADERBOARD_PATH = config.RESULTS_DIR / "subject_leaderboard.csv"
SCORED_RESULTS_PATH      = config.RESULTS_DIR / "scored_results.csv"
FIGURES_DIR              = config.FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Visual Design System
# ---------------------------------------------------------------------------
PALETTE = [
    "#4C9BE8",
    "#E8834C",
    "#4CE8A0",
    "#E84C6B",
    "#B04CE8",
    "#E8D44C",
    "#4CE8E8",
    "#E84CB4",
]

BG_COLOR    = "#0F1117"
PANEL_COLOR = "#1A1D27"
TEXT_COLOR  = "#E8EAF0"
GRID_COLOR  = "#2A2D3A"

METRIC_COLORS = {
    "Exact Match %":    "#4C9BE8",
    "ROUGE-L":          "#4CE8A0",
    "BERT-F1":          "#B04CE8",
    "LLM Score (1-5)":  "#E8834C",
    "LLM Score (0-10)": "#E8D44C",
}


def apply_dark_style() -> None:
    """Configure matplotlib for a premium dark-mode aesthetic."""
    plt.rcParams.update({
        "figure.facecolor":      BG_COLOR,
        "axes.facecolor":        PANEL_COLOR,
        "axes.edgecolor":        GRID_COLOR,
        "axes.labelcolor":       TEXT_COLOR,
        "axes.titlecolor":       TEXT_COLOR,
        "axes.titlesize":        13,
        "axes.titleweight":      "bold",
        "axes.labelsize":        10,
        "axes.grid":             True,
        "axes.axisbelow":        True,
        "grid.color":            GRID_COLOR,
        "grid.linewidth":        0.6,
        "xtick.color":           TEXT_COLOR,
        "ytick.color":           TEXT_COLOR,
        "xtick.labelsize":       9,
        "ytick.labelsize":       9,
        "legend.facecolor":      PANEL_COLOR,
        "legend.edgecolor":      GRID_COLOR,
        "legend.labelcolor":     TEXT_COLOR,
        "legend.fontsize":       9,
        "legend.title_fontsize": 9,
        "text.color":            TEXT_COLOR,
        "figure.dpi":            150,
        "savefig.dpi":           200,
        "savefig.bbox":          "tight",
        "savefig.facecolor":     BG_COLOR,
        "font.family":           "DejaVu Sans",
    })


def savefig(fig: plt.Figure, name: str) -> Path:
    out = FIGURES_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK]  {out}")
    return out


def short_name(model: str, max_len: int = 20) -> str:
    """Shorten a model tag for axis labels."""
    return model if len(model) <= max_len else model[:max_len - 1] + "..."


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_leaderboard() -> pd.DataFrame:
    """Load results/leaderboard.csv."""
    if not LEADERBOARD_PATH.exists():
        sys.exit(
            f"ERROR: {LEADERBOARD_PATH} not found.\n"
            "       Run step3_evaluate.py first."
        )
    df = pd.read_csv(LEADERBOARD_PATH)
    for col in [
        "n_questions", "exact_match_rate",
        "avg_rouge_l", "avg_bert_score_f1",
        "avg_llm_score_1_5", "avg_llm_score_0_10",
        "avg_latency_s", "n_errors",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["model_label"] = df["model"].apply(short_name)
    return df


def load_subject_leaderboard() -> pd.DataFrame:
    """
    Load results/subject_leaderboard.csv directly (written by step3_evaluate.py).
    No re-derivation from scored_results.csv is performed here.
    """
    if not SUBJECT_LEADERBOARD_PATH.exists():
        sys.exit(
            f"ERROR: {SUBJECT_LEADERBOARD_PATH} not found.\n"
            "       Run step3_evaluate.py first."
        )
    if SUBJECT_LEADERBOARD_PATH.stat().st_size == 0:
        sys.exit(
            f"ERROR: {SUBJECT_LEADERBOARD_PATH} exists but is empty.\n"
            "       Re-run step3_evaluate.py to regenerate it."
        )
    df = pd.read_csv(SUBJECT_LEADERBOARD_PATH)
    for col in [
        "n_questions", "exact_match_rate",
        "avg_rouge_l", "avg_bert_score_f1",
        "avg_llm_score_1_5",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["model_label"] = df["model"].apply(short_name)
    return df


def load_scored_results():
    """
    Optionally load results/scored_results.csv.
    Returns None with a warning if the file is absent or empty.
    Charts that need it will be skipped gracefully.
    """
    if not SCORED_RESULTS_PATH.exists():
        print(
            f"  [WARN] {SCORED_RESULTS_PATH} not found - "
            "per-record distribution charts will be skipped."
        )
        return None
    if SCORED_RESULTS_PATH.stat().st_size == 0:
        print(
            f"  [WARN] {SCORED_RESULTS_PATH} is empty - "
            "per-record distribution charts will be skipped."
        )
        return None

    df = pd.read_csv(SCORED_RESULTS_PATH)
    for col in [
        "exact_match", "rouge_l", "bert_score_f1",
        "llm_score_0_10", "llm_score_1_5",
        "latency_s", "prompt_tokens", "completion_tokens", "total_tokens",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["model_label"] = df["model"].apply(short_name)
    return df


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _add_bar_labels(
    ax: plt.Axes,
    bars,
    fmt: str = "{:.2f}",
    color: str = TEXT_COLOR,
    fontsize: int = 8,
    v_offset_frac: float = 0.012,
) -> None:
    """Annotate vertical bar patches with their numeric value."""
    y_max = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + y_max * v_offset_frac,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=fontsize, color=color, fontweight="bold",
        )


def _styled_fig(figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 01 - Overall performance bar chart
#   Metrics: Exact Match %, ROUGE-L, BERT-F1, LLM Score (1-5)
# ---------------------------------------------------------------------------

def plot_overall_score_bars(lb: pd.DataFrame) -> None:
    """
    Grouped bar chart showing Exact Match %, ROUGE-L, BERT-F1, and LLM(1-5)
    side-by-side for every model in the leaderboard.
    """
    models = lb["model_label"].tolist()
    n      = len(models)
    x      = np.arange(n)

    metrics = {}
    metrics["Exact Match %"] = lb["exact_match_rate"].fillna(0) * 100
    if "avg_rouge_l" in lb.columns:
        metrics["ROUGE-L"] = lb["avg_rouge_l"].fillna(0)
    if "avg_bert_score_f1" in lb.columns:
        metrics["BERT-F1"] = lb["avg_bert_score_f1"].fillna(0)
    metrics["LLM Score (1-5)"] = lb["avg_llm_score_1_5"].fillna(0)

    n_metrics = len(metrics)
    width     = min(0.72 / n_metrics, 0.22)
    offsets   = np.linspace(
        -(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics
    ) * width

    fig, ax = _styled_fig(figsize=(max(11, n * 4.5 + 2), 6))

    for (label, values), offset in zip(metrics.items(), offsets):
        color = METRIC_COLORS.get(label, PALETTE[0])
        bars = ax.bar(
            x + offset, values.values, width,
            label=label, color=color,
            alpha=0.88, zorder=3,
            edgecolor=BG_COLOR, linewidth=0.6,
        )
        _add_bar_labels(ax, bars, fmt="{:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Overall Performance Comparison - EduBench-Local", pad=16)
    ax.legend(loc="upper right", framealpha=0.85)

    all_vals = np.concatenate([v.values for v in metrics.values()])
    y_max = float(np.nanmax(all_vals)) if len(all_vals) > 0 else 10.0
    ax.set_ylim(0, y_max * 1.25)

    for ref in [1, 2, 3, 4, 5]:
        ax.axhline(ref, color=GRID_COLOR, lw=0.5, ls=":", zorder=1)

    fig.tight_layout()
    savefig(fig, "01_overall_score_bars.png")


# ---------------------------------------------------------------------------
# Figure 02 - Latency vs Accuracy scatter
# ---------------------------------------------------------------------------

def plot_latency_vs_accuracy(lb: pd.DataFrame) -> None:
    fig, ax = _styled_fig(figsize=(9, 6))

    sizes = (lb["n_questions"] / lb["n_questions"].max() * 600 + 100).values

    for i, (_, row) in enumerate(lb.iterrows()):
        c = PALETTE[i % len(PALETTE)]
        ax.scatter(
            row["avg_latency_s"], row["avg_llm_score_1_5"],
            s=sizes[i], color=c, alpha=0.90, zorder=4,
            edgecolors="white", linewidths=0.9,
        )
        ax.annotate(
            row["model_label"],
            xy=(row["avg_latency_s"], row["avg_llm_score_1_5"]),
            xytext=(7, 7), textcoords="offset points",
            fontsize=8.5, color=c, fontweight="bold",
        )

    ax.set_xlabel("Average Latency (s)")
    ax.set_ylabel("Average LLM Score (1-5)")
    ax.set_title(
        "Latency vs. Accuracy  -  bubble size proportional to N questions", pad=14
    )

    if len(lb) > 1:
        med_x = lb["avg_latency_s"].median()
        med_y = lb["avg_llm_score_1_5"].median()
        ax.axvline(med_x, color=GRID_COLOR, lw=1.2, ls="--", zorder=2,
                   label=f"Median latency ({med_x:.2f}s)")
        ax.axhline(med_y, color=GRID_COLOR, lw=1.2, ls="--", zorder=2,
                   label=f"Median LLM score ({med_y:.2f})")
        ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    savefig(fig, "02_latency_vs_accuracy.png")


# ---------------------------------------------------------------------------
# Figure 03 - Subject heatmap: model x subject, avg LLM(1-5)
# ---------------------------------------------------------------------------

def plot_subject_heatmap(sub_lb: pd.DataFrame) -> None:
    """
    Reads avg_llm_score_1_5 from the pre-loaded subject_leaderboard DataFrame
    and renders a model x subject heatmap - no CSV re-read.
    """
    pivot = sub_lb.pivot_table(
        index="model_label", columns="subject",
        values="avg_llm_score_1_5", aggfunc="mean",
    )
    if pivot.empty:
        print("  !  Skipping heatmap (no subject data).")
        return

    fig_h = max(4, len(pivot) * 1.1 + 2.5)
    fig_w = max(7, len(pivot.columns) * 1.6 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = sns.color_palette("mako", as_cmap=True)
    sns.heatmap(
        pivot,
        ax=ax, cmap=cmap,
        annot=True, fmt=".2f",
        annot_kws={"size": 11, "weight": "bold", "color": TEXT_COLOR},
        linewidths=0.6, linecolor=GRID_COLOR,
        vmin=1, vmax=5,
        cbar_kws={"label": "Avg LLM Score (1-5)", "shrink": 0.8},
    )
    ax.set_title("LLM Score Heatmap - Model x Subject", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.yaxis.label.set_color(TEXT_COLOR)
        cbar.ax.tick_params(colors=TEXT_COLOR)

    fig.tight_layout()
    savefig(fig, "03_subject_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 04 - Subject bar comparison: LLM(1-5) grouped by subject
# ---------------------------------------------------------------------------

def plot_subject_bar_comparison(sub_lb: pd.DataFrame) -> None:
    subjects = sub_lb["subject"].unique()
    models   = sub_lb["model_label"].unique()
    n_sub    = len(subjects)
    n_mod    = len(models)

    if n_sub == 0:
        print("  !  Skipping subject bar comparison (no data).")
        return

    x     = np.arange(n_sub)
    width = min(0.8 / max(n_mod, 1), 0.30)

    fig, ax = _styled_fig(figsize=(max(10, n_sub * 2.8 + 2), 6))

    for mi, model in enumerate(models):
        subset = sub_lb[sub_lb["model_label"] == model]
        vals = []
        for s in subjects:
            match = subset.loc[subset["subject"] == s, "avg_llm_score_1_5"]
            vals.append(float(match.values[0]) if not match.empty else np.nan)

        offset = (mi - n_mod / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=model, color=PALETTE[mi % len(PALETTE)],
            alpha=0.88, zorder=3,
            edgecolor=BG_COLOR, linewidth=0.6,
        )
        _add_bar_labels(ax, bars, fmt="{:.2f}", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=22, ha="right")
    ax.set_ylabel("Avg LLM Score (1-5)")
    ax.set_title("Per-Subject LLM Score Comparison", pad=14)
    ax.set_ylim(0, 5.9)
    for ref in [1, 2, 3, 4, 5]:
        ax.axhline(ref, color=GRID_COLOR, lw=0.5, ls=":", zorder=1)
    if n_mod > 1:
        ax.legend(title="Model", loc="upper right")

    fig.tight_layout()
    savefig(fig, "04_subject_bar_comparison.png")


# ---------------------------------------------------------------------------
# Figure 05 - Exact-match rate horizontal bar
# ---------------------------------------------------------------------------

def plot_exact_match_rate(lb: pd.DataFrame) -> None:
    lb_sorted = lb.sort_values("exact_match_rate", ascending=True)
    fig, ax   = _styled_fig(figsize=(9, max(4, len(lb_sorted) * 0.9 + 2)))

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(lb_sorted))]
    bars   = ax.barh(
        lb_sorted["model_label"],
        lb_sorted["exact_match_rate"] * 100,
        color=colors, alpha=0.88, zorder=3,
        edgecolor=BG_COLOR, linewidth=0.6, height=0.55,
    )

    x_max = max(float(lb_sorted["exact_match_rate"].max()) * 100, 1.0)
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + x_max * 0.018, bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center", ha="left",
            fontsize=9, color=TEXT_COLOR, fontweight="bold",
        )

    ax.set_xlabel("Exact Match Rate (%)")
    ax.set_title("Exact Match Rate per Model", pad=14)
    ax.set_xlim(0, x_max * 1.25 + 1)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    savefig(fig, "05_exact_match_rate.png")


# ---------------------------------------------------------------------------
# Figure 06 - Multi-metric subject heatmap (ROUGE-L, BERT-F1, LLM)
# ---------------------------------------------------------------------------

def plot_subject_multi_metric_heatmap(sub_lb: pd.DataFrame) -> None:
    """
    Side-by-side heatmaps (one panel per metric) for every subject,
    built from the in-memory subject_leaderboard DataFrame.
    """
    metric_defs = [
        ("ROUGE-L",         "avg_rouge_l",         0.0, 1.0, "Blues"),
        ("BERT-F1",         "avg_bert_score_f1",   0.0, 1.0, "Purples"),
        ("LLM Score (1-5)", "avg_llm_score_1_5",   1.0, 5.0, "mako"),
    ]
    # Keep only columns present in the data
    metric_defs = [(lbl, col, vmin, vmax, cm)
                   for lbl, col, vmin, vmax, cm in metric_defs
                   if col in sub_lb.columns]

    if not metric_defs:
        print("  !  Skipping multi-metric heatmap (no numeric metric columns).")
        return

    pivots = [
        (lbl, sub_lb.pivot_table(
            index="model_label", columns="subject",
            values=col, aggfunc="mean",
        ), vmin, vmax, cm)
        for lbl, col, vmin, vmax, cm in metric_defs
    ]

    n_metrics  = len(pivots)
    n_models   = max(len(p[1]) for p in pivots)
    n_subjects = max(len(p[1].columns) for p in pivots)

    fig_h = max(4, n_models * 1.0 + 2.5)
    fig_w = max(8, n_subjects * 1.5 * n_metrics + 2)

    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_w, fig_h), sharey=True)
    if n_metrics == 1:
        axes = [axes]

    for ax, (label, pivot, vmin, vmax, cmap) in zip(axes, pivots):
        sns.heatmap(
            pivot,
            ax=ax, cmap=cmap,
            annot=True, fmt=".2f",
            annot_kws={"size": 9, "weight": "bold"},
            linewidths=0.5, linecolor=GRID_COLOR,
            vmin=vmin, vmax=vmax,
            cbar_kws={"shrink": 0.75},
        )
        ax.set_title(label, fontsize=11, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("" if ax is not axes[0] else "Model")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "Subject-Wise Metric Breakdown - EduBench-Local",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    savefig(fig, "06_subject_metric_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 07 - LLM score distribution (violin + strip)  [needs scored_results]
# ---------------------------------------------------------------------------

def plot_score_distribution(scored: pd.DataFrame) -> None:
    models       = scored["model_label"].unique()
    n            = len(models)
    palette_dict = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    fig, ax = _styled_fig(figsize=(max(8, n * 2.8 + 2), 6))

    if n >= 2:
        sns.violinplot(
            data=scored, x="model_label", y="llm_score_1_5",
            hue="model_label", palette=palette_dict, legend=False, ax=ax,
            inner=None, cut=0, linewidth=0.8, alpha=0.60,
        )

    sns.stripplot(
        data=scored, x="model_label", y="llm_score_1_5",
        hue="model_label", palette=palette_dict, legend=False, ax=ax,
        size=3, alpha=0.45, jitter=True, zorder=3,
    )

    ax.set_xlabel("")
    ax.set_ylabel("LLM Score (1-5)")
    ax.set_title("LLM Score Distribution per Model", pad=14)
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    savefig(fig, "07_score_distribution.png")


# ---------------------------------------------------------------------------
# Figure 08 - Latency distribution box plot  [needs scored_results]
# ---------------------------------------------------------------------------

def plot_latency_distribution(scored: pd.DataFrame) -> None:
    models       = scored["model_label"].unique()
    n            = len(models)
    palette_dict = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    fig, ax = _styled_fig(figsize=(max(8, n * 2.8 + 2), 6))

    sns.boxplot(
        data=scored, x="model_label", y="latency_s",
        hue="model_label", palette=palette_dict, legend=False, ax=ax,
        linewidth=0.9,
        flierprops=dict(
            marker="o", markerfacecolor=PALETTE[3],
            markersize=3, alpha=0.5, linestyle="none",
        ),
    )

    ax.set_xlabel("")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Response Latency Distribution per Model", pad=14)
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    savefig(fig, "08_latency_distribution.png")


# ---------------------------------------------------------------------------
# Figure 09 - Total tokens vs latency scatter  [needs scored_results]
# ---------------------------------------------------------------------------

def plot_tokens_vs_latency(scored: pd.DataFrame) -> None:
    required = {"total_tokens", "latency_s", "model_label"}
    if not required.issubset(scored.columns):
        print("  !  Skipping tokens vs latency (missing columns).")
        return
    df = scored.dropna(subset=["total_tokens", "latency_s"])
    if df.empty:
        print("  !  Skipping tokens vs latency (no data after dropna).")
        return

    models = df["model_label"].unique()
    fig, ax = _styled_fig(figsize=(10, 6))

    for i, model in enumerate(models):
        sub = df[df["model_label"] == model]
        ax.scatter(
            sub["total_tokens"], sub["latency_s"],
            label=model, color=PALETTE[i % len(PALETTE)],
            s=20, alpha=0.55, zorder=3,
        )

    ax.set_xlabel("Total Tokens (prompt + completion)")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Token Count vs. Response Latency", pad=14)
    if len(models) > 1:
        ax.legend(title="Model", loc="upper left")

    fig.tight_layout()
    savefig(fig, "09_tokens_vs_latency.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    apply_dark_style()

    print("=" * 64)
    print("EduBench-Local  |  Step 4: Visualize")
    print("=" * 64)
    print(f"  Leaderboard         : {LEADERBOARD_PATH}")
    print(f"  Subject leaderboard : {SUBJECT_LEADERBOARD_PATH}")
    print(f"  Scored results      : {SCORED_RESULTS_PATH}  (optional)")
    print(f"  Figures dir         : {FIGURES_DIR}")
    print()

    # -- Required: leaderboard + subject leaderboard -----------------------
    print("Loading leaderboard data ...")
    lb     = load_leaderboard()
    sub_lb = load_subject_leaderboard()

    print(f"  Models   : {lb['model'].tolist()}")
    print(f"  Subjects : {sub_lb['subject'].unique().tolist()}")
    print(f"  LB rows  : {len(lb)}  |  Subject-LB rows: {len(sub_lb)}")
    print()

    # -- Optional: per-question scored results for distribution charts -----
    print("Loading per-question scored results (optional) ...")
    scored = load_scored_results()
    if scored is not None:
        print(f"  Loaded {len(scored):,} scored rows.")
    print()

    # -- Generate figures --------------------------------------------------
    print("Generating figures ...")

    # Core charts (always run - require only the two leaderboard CSVs)
    plot_overall_score_bars(lb)
    plot_latency_vs_accuracy(lb)
    plot_subject_heatmap(sub_lb)
    plot_subject_bar_comparison(sub_lb)
    plot_exact_match_rate(lb)
    plot_subject_multi_metric_heatmap(sub_lb)

    # Distribution charts (only when scored_results.csv is available)
    if scored is not None:
        plot_score_distribution(scored)
        plot_latency_distribution(scored)
        plot_tokens_vs_latency(scored)
    else:
        print(
            "  [SKIP] Per-record distribution charts skipped "
            "(scored_results.csv not available)."
        )

    print()
    print(f"All figures saved to: {FIGURES_DIR}")
    print("Step 4 complete.")


if __name__ == "__main__":
    main()
