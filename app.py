# -*- coding: utf-8 -*-
"""
app.py - EduBench-Local Interactive Dashboard
==============================================
A Streamlit web dashboard for the EduBench-Local benchmarking pipeline.

Features:
  - Multi-model evaluation support: interactive dropdown/toggle between
    "Qwen 2.5 3B" and "Google Gemini", or a side-by-side comparison view.
  - Live data loading: dynamically switches between results/scored_results.csv (Qwen)
    and results/gemini_scored_results.csv (Gemini).
  - Visualizations gallery: showcases all 18 unique figures across Qwen (6), Gemini (6), and Comparison (6).
  - Live model playground: test queries in real-time with latency tracking.
  - Auto-launcher: launches Streamlit automatically when run via `python app.py`.

Usage:
  python app.py
  # or: streamlit run app.py
"""

from __future__ import annotations

import io
import os
import sys

# Force UTF-8 encoding on Windows to prevent UnicodeEncodeError in Click/Streamlit
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Auto-launcher: if executed via `python app.py`, launch Streamlit automatically
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _under_streamlit = False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _under_streamlit = (get_script_run_ctx() is not None)
    except Exception:
        pass

    if not _under_streamlit:
        from streamlit.web import cli as stcli
        target_script = os.path.abspath(__file__)
        sys.argv = ["streamlit", "run", target_script] + sys.argv[1:]
        sys.exit(stcli.main())


import time
from pathlib import Path
from collections import defaultdict
from typing import Any

import pandas as pd
import streamlit as st

import config

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EduBench-Local Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — white-based warm cream & crimson design (#FFFAF3, #FFF2DB, #FFE5BF, #F62440)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Core background ── */
    [data-testid="stAppViewContainer"], .main {
        background-color: #FFFAF3 !important;
    }
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background-color: #FFF2DB !important;
        border-right: 1.5px solid #FFE5BF !important;
    }
    [data-testid="stHeader"] {
        background-color: rgba(255, 250, 243, 0.95) !important;
    }

    /* ── Typography ── */
    html, body, [class*="css"], [data-testid="stMarkdownContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #2D1E1E;
    }
    h1 {
        color: #F62440 !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
    }
    h2 {
        color: #1E1B18 !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px !important;
    }
    h3 {
        color: #2D1E1E !important;
        font-weight: 600 !important;
    }
    h4, h5, h6 {
        color: #3D2B24 !important;
        font-weight: 600 !important;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #FFFFFF !important;
        border: 1.5px solid #FFE5BF !important;
        border-radius: 14px !important;
        padding: 16px 20px 14px !important;
        box-shadow: 0 2px 10px rgba(255, 229, 191, 0.5), 0 1px 3px rgba(45, 30, 30, 0.04) !important;
        transition: all 0.2s ease;
    }
    [data-testid="metric-container"]:hover {
        border-color: #F62440 !important;
        box-shadow: 0 4px 16px rgba(246, 36, 64, 0.12) !important;
        transform: translateY(-1px);
    }
    [data-testid="metric-container"] label {
        color: #6E5D53 !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #F62440 !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
        color: #2D1E1E !important;
    }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1.5px solid #FFE5BF;
        background: #FFFFFF;
        box-shadow: 0 2px 8px rgba(255, 229, 191, 0.3);
    }

    /* ── Sidebar nav & radio labels ── */
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #2D1E1E !important;
        padding: 6px 0;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #2D1E1E;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #F62440 0%, #D81A34 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(246, 36, 64, 0.32) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #FF3B56 0%, #F62440 100%) !important;
        box-shadow: 0 6px 20px rgba(246, 36, 64, 0.48) !important;
        transform: translateY(-1px);
        color: #FFFFFF !important;
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Select / text inputs ── */
    .stSelectbox > div > div,
    .stTextArea > div > div,
    .stTextInput > div > div {
        background: #FFFFFF !important;
        border: 1.5px solid #FFE5BF !important;
        border-radius: 10px !important;
        color: #2D1E1E !important;
        box-shadow: 0 1px 4px rgba(255, 229, 191, 0.3) !important;
    }
    .stSelectbox > div > div:focus-within,
    .stTextArea > div > div:focus-within,
    .stTextInput > div > div:focus-within {
        border-color: #F62440 !important;
        box-shadow: 0 0 0 3px rgba(246, 36, 64, 0.15) !important;
    }

    /* ── Code / response box ── */
    .response-box {
        background: #FFFFFF;
        border: 1.5px solid #FFE5BF;
        border-radius: 12px;
        padding: 20px 24px;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #1E1B18;
        white-space: pre-wrap;
        max-height: 420px;
        overflow-y: auto;
        box-shadow: 0 2px 10px rgba(255, 229, 191, 0.35);
    }
    .latency-badge {
        display: inline-block;
        background: #FFF2DB;
        color: #F62440;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 10px;
        border: 1.5px solid #FFE5BF;
    }
    .error-box {
        background: #FFF5F5;
        border: 1.5px solid #F62440;
        border-radius: 12px;
        padding: 16px 20px;
        color: #B91C1C;
        font-size: 0.9rem;
    }

    /* ── Section divider ── */
    .section-divider {
        border: none;
        height: 1.5px;
        background: linear-gradient(90deg, transparent, #FFE5BF, transparent);
        margin: 28px 0;
    }

    /* ── Page hero ── */
    .page-hero {
        background: linear-gradient(135deg, #FFF2DB 0%, #FFE5BF 100%);
        border: 1.5px solid #FFE5BF;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 3px 12px rgba(255, 229, 191, 0.45);
    }
    .page-hero h1 {
        margin: 0 0 6px 0;
        font-size: 1.85rem;
        color: #F62440 !important;
    }
    .page-hero p {
        color: #6E5D53;
        margin: 0;
        font-size: 0.95rem;
        font-weight: 500;
    }

    /* ── Model badge pill ── */
    .model-pill {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .pill-qwen {
        background: #FFF2DB;
        color: #2D1E1E;
        border: 1.5px solid #FFE5BF;
    }
    .pill-gemini {
        background: #FFF2DB;
        color: #F62440;
        border: 1.5px solid #FFE5BF;
    }
    .pill-compare {
        background: #FFF2DB;
        color: #F62440;
        border: 1.5px solid #F62440;
    }

    /* ── Figure card ── */
    .fig-card {
        background: #FFFFFF;
        border: 1.5px solid #FFE5BF;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 2px 10px rgba(255, 229, 191, 0.4);
        transition: all 0.2s ease;
    }
    .fig-card:hover {
        border-color: #F62440;
        box-shadow: 0 6px 24px rgba(246, 36, 64, 0.16);
        transform: translateY(-2px);
    }

    /* ── Expander ── */
    div[data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1.5px solid #FFE5BF;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers & Data Loaders
# ---------------------------------------------------------------------------
RESULTS_DIR = config.RESULTS_DIR
FIGURES_DIR = config.FIGURES_DIR

DATASET_NAME_MAP = {
    "science":                     "SciQ",
    "general_science":             "OpenBookQA",
    "science_challenge":           "ARC-Challenge",
    "reading_comprehension":       "RACE",
    "reading_comprehension_squad": "SQuAD v1.1",
}
SUBJECT_NAME_MAP = {v: k for k, v in DATASET_NAME_MAP.items()}
DATASET_ORDER = ["SciQ", "OpenBookQA", "ARC-Challenge", "RACE", "SQuAD v1.1"]
MODELS = ["Qwen 2.5 3B", "Google Gemini", "⚔️ Head-to-Head Comparison"]

FIGURE_TITLES = {
    # ── Qwen 2.5 (3B) Individual Visualizations ──
    "qwen_bars.png": "Qwen 2.5 (3B) — Multi-Metric Performance Across Datasets",
    "qwen_radar.png": "Qwen 2.5 (3B) — Multi-Metric Radar Profile Across Datasets",
    "qwen_boxplots.png": "Qwen 2.5 (3B) — Score Distributions by Dataset (Boxplots)",
    "qwen_heatmap.png": "Qwen 2.5 (3B) — Metric Score Heatmap (Dataset × Metric)",
    "qwen_latency.png": "Qwen 2.5 (3B) — Generation Latency by Dataset",
    "qwen_correlations.png": "Qwen 2.5 (3B) — Metric Correlation Analysis (ROUGE-L, BERT-F1, Judge)",

    # ── Google Gemini Individual Visualizations ──
    "gemini_bars.png": "Google Gemini — Multi-Metric Performance Across Datasets",
    "gemini_radar.png": "Google Gemini — Multi-Metric Radar Profile Across Datasets",
    "gemini_boxplots.png": "Google Gemini — Score Distributions by Dataset (Boxplots)",
    "gemini_heatmap.png": "Google Gemini — Metric Score Heatmap (Dataset × Metric)",
    "gemini_latency.png": "Google Gemini — Generation Latency by Dataset",
    "gemini_correlations.png": "Google Gemini — Metric Correlation Analysis (ROUGE-L, Token F1, Char Sim)",

    # ── Head-to-Head Comparison Visualizations ──
    "comparison_bars.png": "Head-to-Head Comparison — Multi-Metric Horizontal Bars Across 5 Datasets",
    "comparison_scorecard.png": "Head-to-Head Comparison — Executive Comparative Scorecard & Win Matrix",
    "comparison_f1.png": "Head-to-Head Comparison — Dataset F1 Performance & Category Winner",
    "comparison_latency.png": "Head-to-Head Comparison — Generation Latency Dual-Panel Comparison",
    "comparison_radar.png": "Head-to-Head Comparison — Dual-Model Comparative Radar Footprint",
    "comparison_heatmap.png": "Head-to-Head Comparison — Side-by-Side Normalized Metric Matrix Heatmap",
}



@st.cache_data(ttl=30)
def load_leaderboard() -> pd.DataFrame | None:
    p = RESULTS_DIR / "leaderboard.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    num = ["n_questions", "exact_match_rate", "avg_rouge_l", "avg_bert_score_f1",
           "avg_llm_score_1_5", "avg_llm_score_0_10", "avg_latency_s", "n_errors"]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_subject_leaderboard() -> pd.DataFrame | None:
    p = RESULTS_DIR / "subject_leaderboard.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ["n_questions", "exact_match_rate", "avg_rouge_l", "avg_bert_score_f1",
              "avg_llm_score_1_5", "avg_llm_score_0_10"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_qwen_scored() -> pd.DataFrame | None:
    """Load Qwen per-question scored results from results/scored_results.csv."""
    p = RESULTS_DIR / "scored_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ["exact_match", "rouge_l", "bert_score_f1", "llm_score_0_10", "llm_score_1_5",
              "latency_s", "prompt_tokens", "completion_tokens", "total_tokens"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_gemini_scored() -> pd.DataFrame | None:
    """Load Gemini per-question scored results from results/gemini_scored_results.csv."""
    p = RESULTS_DIR / "gemini_scored_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ["exact_match", "token_f1", "token_precision", "token_recall",
              "rouge_l", "char_similarity", "contains_match"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_gemini_leaderboard() -> pd.DataFrame | None:
    """Load Gemini aggregated leaderboard from results/gemini_leaderboard.csv."""
    p = RESULTS_DIR / "gemini_leaderboard.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ["n_questions", "exact_match_rate", "avg_token_f1", "avg_rouge_l",
              "avg_char_similarity", "contains_match_rate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def missing_data_warning(label: str) -> None:
    st.warning(
        f"**{label}** not found. "
        "Run the evaluation script to generate it.",
        icon="⚠️",
    )


# ---------------------------------------------------------------------------
# Global Session State & Synchronization
# ---------------------------------------------------------------------------
if "active_model" not in st.session_state:
    st.session_state["active_model"] = "Qwen 2.5 3B"
if "sb_model_key" not in st.session_state:
    st.session_state["sb_model_key"] = st.session_state["active_model"]
if "main_model_key" not in st.session_state:
    st.session_state["main_model_key"] = st.session_state["active_model"]

def on_sidebar_model_change():
    val = st.session_state["sb_model_key"]
    st.session_state["active_model"] = val
    st.session_state["main_model_key"] = val

def on_main_model_change():
    val = st.session_state["main_model_key"]
    st.session_state["active_model"] = val
    st.session_state["sb_model_key"] = val

# ---------------------------------------------------------------------------
# Sidebar Navigation & Model Selector
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 16px 0 10px;'>
            <span style='font-size:2.4rem;'>🎓</span><br>
            <span style='font-size:1.15rem; font-weight:800;
                         background: linear-gradient(90deg, #F62440, #D81A34);
                         -webkit-background-clip:text;
                         -webkit-text-fill-color:transparent;'>
                EduBench-Local
            </span><br>
            <span style='font-size:0.75rem; color:#6E5D53; font-weight:600;'>
                Multi-Model LLM Benchmark
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border:none; border-top:1.5px solid #FFE5BF; margin:4px 0 14px;'>", unsafe_allow_html=True)

    # ── Interactive Model Selection ──
    st.markdown("**🤖 Active Evaluation Model**")
    st.selectbox(
        "Model Selection",
        MODELS,
        key="sb_model_key",
        on_change=on_sidebar_model_change,
        label_visibility="collapsed",
        help="Select the model to inspect its detailed evaluation metrics and generated answers."
    )
    active_model = st.session_state["active_model"]

    if active_model == "Qwen 2.5 3B":
        st.caption("🔴 **Qwen 2.5 (3B)** · Local Ollama · 750 questions evaluated")
    elif active_model == "Google Gemini":
        st.caption("🟠 **Google Gemini** · Cloud Flash API · 5-question micro-sample")
    else:
        st.caption("⚔️ **Head-to-Head** · Comparative benchmark across 5 datasets")

    st.markdown("<hr style='border:none; border-top:1.5px solid #FFE5BF; margin:14px 0 16px;'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊  Leaderboard & Analytics", "🖼️  Visualizations", "🧪  Live Model Playground"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border:none; border-top:1.5px solid #FFE5BF; margin:16px 0 12px;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.72rem; color:#6E5D53; text-align:center;'>"
        "EduBench-Local Evaluation Suite<br>"
        f"Data dir: <code style='color:#F62440; background:#FFF2DB; padding:2px 6px; border-radius:4px; border:1px solid #FFE5BF;'>{RESULTS_DIR.name}/</code>"
        "</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 1 — Leaderboard & Analytics (Interactive Model Switching)
# ===========================================================================
if page == "📊  Leaderboard & Analytics":
    active_model = st.session_state["active_model"]

    pill_cls = (
        "pill-qwen" if active_model == "Qwen 2.5 3B"
        else ("pill-gemini" if active_model == "Google Gemini" else "pill-compare")
    )

    st.markdown(
        f"""
        <div class='page-hero'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <h1>📊 Benchmark Leaderboard &amp; Analytics</h1>
                    <p>Evaluating educational QA performance across SciQ, OpenBookQA, ARC-Challenge, RACE, and SQuAD v1.1</p>
                </div>
                <div>
                    <span class='model-pill {pill_cls}'>
                        {active_model}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Top interactive model switcher tabs ──
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.radio(
            "Select Model View",
            MODELS,
            horizontal=True,
            key="main_model_key",
            on_change=on_main_model_change,
        )

    model_view = st.session_state["active_model"]
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # VIEW 1: Qwen 2.5 3B
    # -----------------------------------------------------------------------
    if model_view == "Qwen 2.5 3B":
        lb = load_leaderboard()
        sub_lb = load_subject_leaderboard()
        qwen_scored = load_qwen_scored()

        if lb is None or lb.empty:
            missing_data_warning("results/leaderboard.csv")
            st.stop()

        qwen_row = lb.iloc[0]
        qwen_em = float(qwen_row["exact_match_rate"]) * 100
        qwen_rl = float(qwen_row.get("avg_rouge_l", 0.137))
        qwen_f1 = float(qwen_row.get("avg_bert_score_f1", 0.863))
        qwen_llm = float(qwen_row.get("avg_llm_score_1_5", 4.113))
        qwen_lat = float(qwen_row.get("avg_latency_s", 3.69))
        qwen_n = int(qwen_row["n_questions"])

        # ── 1. Top Metric Cards (Performance Spotlight) ──
        st.subheader("🔴 Qwen 2.5 (3B) Performance Spotlight")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("🤖 Architecture", "Qwen 2.5 (3B)")
        c2.metric("✅ Exact Match", f"{qwen_em:.1f}%")
        c3.metric("📈 Mean ROUGE-L", f"{qwen_rl:.3f}")
        c4.metric("🎯 BERTScore F1", f"{qwen_f1:.3f}")
        c5.metric("⭐ Judge Score", f"{qwen_llm:.2f}", delta="1–5 Scale", delta_color="off")
        c6.metric("⏱️ Mean Latency", f"{qwen_lat:.2f}s", delta=f"{qwen_n:,} Qs", delta_color="off")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── 2. Per-Dataset Performance Breakdown ──
        if sub_lb is not None and not sub_lb.empty:
            st.subheader("📊 Individual Dataset Metric Breakdown")
            col_l, col_r = st.columns([3, 2], gap="large")

            with col_l:
                disp_sub = sub_lb.copy()
                disp_sub["Dataset"] = disp_sub["subject"].map(DATASET_NAME_MAP).fillna(disp_sub["subject"])
                disp_sub["sort_order"] = disp_sub["Dataset"].map(lambda x: DATASET_ORDER.index(x) if x in DATASET_ORDER else 99)
                disp_sub = disp_sub.sort_values("sort_order").drop(columns=["sort_order"])

                disp_sub["Questions (N)"] = disp_sub["n_questions"].astype(int)
                disp_sub["Exact Match %"] = (disp_sub["exact_match_rate"] * 100).map("{:.2f}%".format)
                disp_sub["ROUGE-L"] = disp_sub.get("avg_rouge_l", 0).map("{:.4f}".format)
                disp_sub["BERTScore F1"] = disp_sub.get("avg_bert_score_f1", 0).map("{:.4f}".format)
                disp_sub["Judge Score (1–5)"] = disp_sub.get("avg_llm_score_1_5", 0).map("{:.4f}".format)

                show_cols = ["Dataset", "Questions (N)", "Exact Match %", "ROUGE-L", "BERTScore F1", "Judge Score (1–5)"]
                st.dataframe(
                    disp_sub[show_cols],
                    use_container_width=True,
                    hide_index=True,
                )

            with col_r:
                chart_sub = sub_lb.copy()
                chart_sub["Dataset"] = chart_sub["subject"].map(DATASET_NAME_MAP).fillna(chart_sub["subject"])
                chart_sub["sort_order"] = chart_sub["Dataset"].map(lambda x: DATASET_ORDER.index(x) if x in DATASET_ORDER else 99)
                chart_sub = chart_sub.sort_values("sort_order")
                chart_df = chart_sub.set_index("Dataset")[["avg_rouge_l", "exact_match_rate"]]
                chart_df.columns = ["ROUGE-L", "Exact Match Rate"]
                st.markdown("**Dataset Metric Spread**")
                st.bar_chart(chart_df, height=270)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── 3. Qwen Individual Visualizations (Exactly 6 Files) ──
        st.subheader("📈 Qwen 2.5 (3B) Individual Visualizations")
        st.caption("Clean white publication-grade charts evaluating Qwen 2.5 (3B) exclusively across all educational benchmarks")

        f_qwen_bars = FIGURES_DIR / "qwen_bars.png"
        f_qwen_heat = FIGURES_DIR / "qwen_heatmap.png"
        f_qwen_radar = FIGURES_DIR / "qwen_radar.png"
        f_lat_qwen = FIGURES_DIR / "qwen_latency.png"
        f_qwen_dist = FIGURES_DIR / "qwen_boxplots.png"
        f_qwen_corr = FIGURES_DIR / "qwen_correlations.png"

        cq1, cq2 = st.columns(2, gap="medium")
        with cq1:
            if f_qwen_bars.exists():
                st.markdown("**Multi-Metric Performance Across Datasets (ROUGE-L, Judge Scores, EM)**")
                st.image(str(f_qwen_bars), caption="Exact Match %, ROUGE-L Overlap, and Judge Score (1–5) by Dataset", use_container_width=True)
        with cq2:
            if f_qwen_heat.exists():
                st.markdown("**Metric Score Heatmap (Dataset × Metric)**")
                st.image(str(f_qwen_heat), caption="Normalized score heatmap: ROUGE-L, BERTScore F1, and Judge Quality", use_container_width=True)

        cq3, cq4 = st.columns(2, gap="medium")
        with cq3:
            if f_qwen_radar.exists():
                st.markdown("**Multi-Metric Radar Profile**")
                st.image(str(f_qwen_radar), caption="Radar footprint across SciQ, OpenBookQA, ARC, RACE, SQuAD", use_container_width=True)
        with cq4:
            if f_lat_qwen.exists():
                st.markdown("**Generation Latency by Dataset**")
                st.image(str(f_lat_qwen), caption="Average generation latency (seconds) ± Std Dev error bars", use_container_width=True)

        cq5, cq6 = st.columns(2, gap="medium")
        with cq5:
            if f_qwen_dist.exists():
                st.markdown("**Score Distributions by Dataset (Boxplots)**")
                st.image(str(f_qwen_dist), caption="Boxplot spreads for ROUGE-L, BERTScore F1, and Judge Score (1–5)", use_container_width=True)
        with cq6:
            if f_qwen_corr.exists():
                st.markdown("**Metric Correlation Analysis**")
                st.image(str(f_qwen_corr), caption="Scatter & linear regressions comparing ROUGE-L, BERT-F1, and Judge Score", use_container_width=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── 3. Detailed Generated Answers Explorer ──
        if qwen_scored is not None and not qwen_scored.empty:
            st.subheader("Detailed Generated Answers Explorer (Qwen)")

            f1, f2 = st.columns([2, 2])
            with f1:
                sel_ds = st.selectbox("Filter by Dataset", ["All"] + DATASET_ORDER, key="qwen_ds_filter")
            with f2:
                search_query = st.text_input("🔍 Search in Questions / Answers", "", key="qwen_search")

            filtered = qwen_scored.copy()
            filtered["Dataset"] = filtered["subject"].map(DATASET_NAME_MAP).fillna(filtered["subject"])
            if sel_ds != "All":
                filtered = filtered[filtered["Dataset"] == sel_ds]
            if search_query.strip():
                q_low = search_query.strip().lower()
                filtered = filtered[
                    filtered["question"].str.lower().str.contains(q_low, na=False) |
                    filtered["student_answer"].str.lower().str.contains(q_low, na=False)
                ]

            st.caption(f"Showing **{len(filtered):,}** of **{len(qwen_scored):,}** answers")

            disp_cols = ["id", "Dataset", "subject", "question", "reference_answer", "student_answer",
                         "exact_match", "rouge_l", "llm_score_1_5", "latency_s"]
            avail_cols = [c for c in disp_cols if c in filtered.columns]
            st.dataframe(
                filtered[avail_cols].rename(columns={
                    "id": "ID", "Dataset": "Dataset", "subject": "Internal Subject", "question": "Question",
                    "reference_answer": "Ground Truth", "student_answer": "Model Answer",
                    "exact_match": "Exact Match", "rouge_l": "ROUGE-L", "llm_score_1_5": "Quality Score",
                    "latency_s": "Latency (s)"
                }),
                use_container_width=True,
                hide_index=True,
                height=380,
            )

            # Individual Question Deep-Dive Expanders
            st.markdown("#### 🔍 Individual Question Deep-Dive")
            deep_dive_samples = filtered.head(5)
            if len(filtered) > 5:
                st.caption(f"Displaying top {len(deep_dive_samples)} inspection cards from current filter")
            for _, row in deep_dive_samples.iterrows():
                em_status = "✅ Exact Match (1.0)" if row.get("exact_match") == 1 else "❌ Mismatch (0.0)"
                ds_name = row.get("Dataset") or DATASET_NAME_MAP.get(row.get("subject"), row.get("subject", "Item"))
                with st.expander(f"{ds_name} — {row['id']} [{em_status}]", expanded=False):
                    st.markdown(f"**Question**: {row['question']}")
                    st.markdown(f"**Ground Truth Reference**: `{row['reference_answer']}`")
                    st.markdown(f"**Qwen Answer**: `{row['student_answer']}`")
                    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                    m_c1.metric("Exact Match", int(row.get("exact_match", 0)))
                    m_c2.metric("ROUGE-L", f"{row.get('rouge_l', 0):.3f}")
                    m_c3.metric("Quality Score (LLM)", f"{row.get('llm_score_1_5', 0):.2f}")
                    lat = row.get("latency_s")
                    m_c4.metric("Latency", f"{lat:.2f}s" if pd.notna(lat) else "—")

    # -----------------------------------------------------------------------
    # VIEW 2: Google Gemini
    # -----------------------------------------------------------------------
    elif model_view == "Google Gemini":
        gem_scored = load_gemini_scored()
        gem_lb = load_gemini_leaderboard()

        if gem_scored is None or gem_scored.empty:
            missing_data_warning("results/gemini_scored_results.csv")
            st.info("Run `python gemini_evaluate.py` followed by `python gemini_evaluate_metrics.py` to generate Gemini scores.")
            st.stop()

        n_gem = len(gem_scored)
        gem_em = float(gem_scored["exact_match"].mean()) * 100
        gem_rl = float(gem_scored["rouge_l"].mean())
        gem_f1 = float(gem_scored["token_f1"].mean())
        gem_sim = float(gem_scored["char_similarity"].mean()) if "char_similarity" in gem_scored.columns else 0.570
        gem_lat = float(gem_scored["latency_s"].mean()) if "latency_s" in gem_scored.columns else 1.03

        # ── 1. Top Metric Cards (Performance Spotlight) ──
        st.subheader("🟠 Google Gemini Performance Spotlight")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("🤖 Architecture", "Gemini (Flash)")
        c2.metric("✅ Exact Match", f"{gem_em:.1f}%")
        c3.metric("📈 Mean ROUGE-L", f"{gem_rl:.3f}")
        c4.metric("🎯 Token F1", f"{gem_f1:.3f}")
        c5.metric("🔤 Char Similarity", f"{gem_sim:.3f}")
        c6.metric("⏱️ Mean Latency", f"{gem_lat:.2f}s", delta=f"{n_gem} Qs sample", delta_color="off")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── 2. Per-Dataset Performance Breakdown ──
        if gem_lb is not None and not gem_lb.empty:
            st.subheader("📊 Individual Dataset Metric Breakdown")
            col_gl, col_gr = st.columns([3, 2], gap="large")

            with col_gl:
                disp_glb = gem_lb.copy()
                disp_glb["sort_order"] = disp_glb["dataset"].map(lambda x: DATASET_ORDER.index(x) if x in DATASET_ORDER else 99)
                disp_glb = disp_glb.sort_values("sort_order").drop(columns=["sort_order"])

                disp_glb["Questions (N)"] = disp_glb["n_questions"].astype(int)
                disp_glb["Exact Match %"] = (disp_glb["exact_match_rate"] * 100).map("{:.2f}%".format)
                disp_glb["ROUGE-L"] = disp_glb["avg_rouge_l"].map("{:.4f}".format)
                disp_glb["Token F1"] = disp_glb["avg_token_f1"].map("{:.4f}".format)
                disp_glb["Char Similarity"] = disp_glb["avg_char_similarity"].map("{:.4f}".format)

                show_gcols = ["dataset", "Questions (N)", "Exact Match %", "ROUGE-L", "Token F1", "Char Similarity"]
                st.dataframe(
                    disp_glb[show_gcols].rename(columns={"dataset": "Dataset"}),
                    use_container_width=True,
                    hide_index=True,
                )

            with col_gr:
                chart_glb = gem_lb.copy()
                chart_glb["sort_order"] = chart_glb["dataset"].map(lambda x: DATASET_ORDER.index(x) if x in DATASET_ORDER else 99)
                chart_glb = chart_glb.sort_values("sort_order")
                chart_gdf = chart_glb.set_index("dataset")[["avg_rouge_l", "exact_match_rate"]]
                chart_gdf.columns = ["ROUGE-L Overlap", "Exact Match Rate"]
                st.markdown("**Dataset Metric Spread**")
                st.bar_chart(chart_gdf, height=270)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── 3. Gemini Individual Evaluation Charts (Exactly 6 Files) ──
        st.subheader("📈 Google Gemini Individual Visualizations")
        st.caption("Clean white publication-grade charts evaluating Google Gemini Flash API across all educational benchmarks")

        f_gem_bars = FIGURES_DIR / "gemini_bars.png"
        f_gem_heat = FIGURES_DIR / "gemini_heatmap.png"
        f_gem_radar = FIGURES_DIR / "gemini_radar.png"
        f_lat_gemini = FIGURES_DIR / "gemini_latency.png"
        f_gem_dist = FIGURES_DIR / "gemini_boxplots.png"
        f_gem_corr = FIGURES_DIR / "gemini_correlations.png"

        cg1, cg2 = st.columns(2, gap="medium")
        with cg1:
            if f_gem_bars.exists():
                st.markdown("**Multi-Metric Performance Across Datasets (Exact Match, Token F1, ROUGE-L)**")
                st.image(str(f_gem_bars), caption="Horizontal bar charts showing Gemini accuracy per dataset", use_container_width=True)
        with cg2:
            if f_gem_heat.exists():
                st.markdown("**Metric Score Heatmap (Dataset × Metric)**")
                st.image(str(f_gem_heat), caption="Normalized score heatmap: ROUGE-L, Token F1, Char Similarity, EM", use_container_width=True)

        cg3, cg4 = st.columns(2, gap="medium")
        with cg3:
            if f_gem_radar.exists():
                st.markdown("**Multi-Metric Radar Profile**")
                st.image(str(f_gem_radar), caption="Radar footprint across SciQ, OpenBookQA, ARC, RACE, SQuAD", use_container_width=True)
        with cg4:
            if f_lat_gemini.exists():
                st.markdown("**Generation Latency by Dataset**")
                st.image(str(f_lat_gemini), caption="Average API latency (seconds) ± Std Dev error bars", use_container_width=True)

        cg5, cg6 = st.columns(2, gap="medium")
        with cg5:
            if f_gem_dist.exists():
                st.markdown("**Score Distributions by Dataset (Boxplots)**")
                st.image(str(f_gem_dist), caption="Boxplot & stripplot spreads for ROUGE-L, Token F1, and Character Similarity", use_container_width=True)
        with cg6:
            if f_gem_corr.exists():
                st.markdown("**Metric Correlation Analysis**")
                st.image(str(f_gem_corr), caption="Scatter & linear regressions between ROUGE-L, Token F1, and Character Similarity", use_container_width=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── 3. Detailed Generated Answers Explorer ──
        st.subheader("Detailed Generated Answers Explorer (Gemini)")

        fg1, fg2 = st.columns([2, 2])
        with fg1:
            sel_ds = st.selectbox("Filter by Dataset", ["All"] + DATASET_ORDER, key="gem_ds_filter")
        with fg2:
            gem_search = st.text_input("🔍 Search in Questions / Answers", "", key="gem_search")

        g_filtered = gem_scored.copy()
        g_filtered["Dataset"] = g_filtered["source_dataset"]
        if sel_ds != "All":
            g_filtered = g_filtered[g_filtered["Dataset"] == sel_ds]
        if gem_search.strip():
            g_low = gem_search.strip().lower()
            g_filtered = g_filtered[
                g_filtered["question"].str.lower().str.contains(g_low, na=False) |
                g_filtered["gemini_answer"].str.lower().str.contains(g_low, na=False)
            ]

        st.caption(f"Showing **{len(g_filtered):,}** of **{len(gem_scored):,}** answers")

        g_disp_cols = ["id", "Dataset", "subject", "question", "reference_answer", "gemini_answer",
                       "exact_match", "rouge_l", "token_f1", "model"]
        avail_gcols = [c for c in g_disp_cols if c in g_filtered.columns]

        st.dataframe(
            g_filtered[avail_gcols].rename(columns={
                "id": "ID", "Dataset": "Dataset", "subject": "Internal Subject", "question": "Question",
                "reference_answer": "Ground Truth", "gemini_answer": "Model Answer",
                "exact_match": "Exact Match", "rouge_l": "ROUGE-L", "token_f1": "Quality Score",
                "model": "Model Used"
            }),
            use_container_width=True,
            hide_index=True,
            height=380,
        )

        # Individual Question Deep-Dive Expanders
        st.markdown("#### 🔍 Individual Question Deep-Dive")
        for _, row in g_filtered.iterrows():
            em_status = "✅ Exact Match (1.0)" if row["exact_match"] == 1 else "❌ Mismatch (0.0)"
            ds_name = row.get("Dataset") or row.get("source_dataset", "Item")
            with st.expander(f"{ds_name} — {row['id']} [{em_status}]", expanded=False):
                st.markdown(f"**Question**: {row['question']}")
                if row.get("context") and pd.notna(row["context"]):
                    st.caption(f"**Passage Context**: {str(row['context'])[:300]}...")
                st.markdown(f"**Ground Truth Reference**: `{row['reference_answer']}`")
                st.markdown(f"**Gemini Answer**: `{row['gemini_answer']}`")
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric("Exact Match", int(row.get("exact_match", 0)))
                m_c2.metric("ROUGE-L", f"{row.get('rouge_l', 0):.3f}")
                m_c3.metric("Quality Score (F1)", f"{row.get('token_f1', 0):.3f}")
                m_c4.metric("Char Similarity", f"{row.get('char_similarity', 0):.3f}")

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # VIEW 3: Dedicated Head-to-Head Comparison (All Comparative/Versus Visualizations)
    # -----------------------------------------------------------------------
    else:
        lb = load_leaderboard()
        sub_lb = load_subject_leaderboard()
        gem_scored = load_gemini_scored()
        qwen_scored = load_qwen_scored()

        st.subheader("⚔️ Head-to-Head Comparison: Qwen 2.5 (3B) vs. Google Gemini")
        st.caption("Direct comparative benchmark contrasting local on-device inference against cloud frontier API")

        # Top comparative spotlight cards
        qwen_em = float(lb["exact_match_rate"].iloc[0])*100 if lb is not None and not lb.empty else 2.9
        qwen_rl = float(lb["avg_rouge_l"].iloc[0]) if lb is not None and not lb.empty else 0.137
        gem_em  = float(gem_scored["exact_match"].mean())*100 if gem_scored is not None else 40.0
        gem_rl  = float(gem_scored["rouge_l"].mean()) if gem_scored is not None else 0.533

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Exact Match Rate", f"Qwen: {qwen_em:.1f}%", f"Gemini: {gem_em:.1f}%", delta_color="normal")
        k2.metric("Mean ROUGE-L Overlap", f"Qwen: {qwen_rl:.3f}", f"Gemini: {gem_rl:.3f}", delta_color="normal")
        k3.metric("Deployment Mode", "Qwen: Local Ollama", "Gemini: Cloud Flash API")
        k4.metric("Inference Privacy", "Qwen: 100% On-Device", "Gemini: API Cloud")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── 5 Datasets Comparison Table ──
        st.subheader("📊 Dataset-by-Dataset Benchmark Performance Matrix")

        comp_rows = []
        for ds_name in DATASET_ORDER:
            s = SUBJECT_NAME_MAP.get(ds_name, ds_name)

            # Qwen metrics
            q_match = sub_lb.loc[sub_lb["subject"] == s] if sub_lb is not None else None
            q_em = float(q_match["exact_match_rate"].values[0])*100 if q_match is not None and not q_match.empty else 0.0
            q_rl = float(q_match["avg_rouge_l"].values[0]) if q_match is not None and not q_match.empty else 0.0

            # Gemini metrics
            if gem_scored is not None:
                g_match = gem_scored[gem_scored["subject"] == s]
                g_em = float(g_match["exact_match"].mean())*100 if not g_match.empty else 0.0
                g_rl = float(g_match["rouge_l"].mean()) if not g_match.empty else 0.0
            else:
                g_em, g_rl = 0.0, 0.0

            winner = "Gemini ★" if g_em > q_em or g_rl > q_rl else ("Qwen ★" if q_em > g_em or q_rl > g_rl else "Parity")

            comp_rows.append({
                "Dataset": ds_name,
                "Qwen EM %": f"{q_em:.1f}%",
                "Gemini EM %": f"{g_em:.1f}%",
                "Qwen ROUGE-L": f"{q_rl:.3f}",
                "Gemini ROUGE-L": f"{g_rl:.3f}",
                "Category Winner": winner
            })

        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── Dedicated Suite of Comparative Visualizations (Exactly 6 Files) ──
        st.subheader("⚔️ Head-to-Head Comparative Visualizations Suite")
        st.caption("Publication-grade charts directly contrasting Qwen 2.5 (3B) and Google Gemini")

        f_exec = FIGURES_DIR / "comparison_scorecard.png"
        f_comp_bars = FIGURES_DIR / "comparison_bars.png"
        f_comp_heat = FIGURES_DIR / "comparison_heatmap.png"
        f_comp_f1 = FIGURES_DIR / "comparison_f1.png"
        f_comp_radar = FIGURES_DIR / "comparison_radar.png"
        f_lat_comp = FIGURES_DIR / "comparison_latency.png"

        # 1. Executive Scorecard & Win Matrix
        if f_exec.exists():
            st.markdown("#### 🏆 Executive Comparative Scorecard & Win Matrix")
            st.image(str(f_exec), caption="Figure 1: Overall KPIs and Per-Dataset Head-to-Head Winner Breakdown", use_container_width=True)
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # 2. Side-by-side: Multi-Metric Comparison Bars & Comparative Heatmap
        c_v1, c_v2 = st.columns(2, gap="medium")
        with c_v1:
            if f_comp_bars.exists():
                st.markdown("#### 📊 Comparative Horizontal Bars Across 5 Datasets")
                st.image(str(f_comp_bars), caption="Figure 2: Head-to-Head Multi-Metric Comparison (Exact Match, ROUGE-L, F1)", use_container_width=True)
        with c_v2:
            if f_comp_heat.exists():
                st.markdown("#### 🗺️ Head-to-Head Score Heatmap")
                st.image(str(f_comp_heat), caption="Figure 3: Side-by-Side Normalized Metric Matrix (Qwen vs. Gemini)", use_container_width=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # 3. Side-by-side: Dataset F1 Comparison & Comparative Radar Footprint
        c_v3, c_v4 = st.columns(2, gap="medium")
        with c_v3:
            if f_comp_f1.exists():
                st.markdown("#### 🎯 Dataset F1 Performance & Category Winner")
                st.image(str(f_comp_f1), caption="Figure 4: Head-to-Head Dataset F1 Comparison with Win Badges", use_container_width=True)
        with c_v4:
            if f_comp_radar.exists():
                st.markdown("#### 🕸️ Dual-Model Comparative Radar Footprint")
                st.image(str(f_comp_radar), caption="Figure 5: Comparative Radar Footprint across SciQ, OpenBookQA, ARC, RACE, SQuAD", use_container_width=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # 4. Dual-Panel Generation Latency Comparison
        if f_lat_comp.exists():
            st.markdown("#### ⚡ Generation Latency: Head-to-Head Dual Panel Comparison")
            st.image(str(f_lat_comp), caption="Figure 6: Average Generation Latency (seconds) — Local Ollama vs. Cloud Flash API", use_container_width=True)
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── 5. Question-by-Question Head-to-Head Answer Explorer ──
        if gem_scored is not None and not gem_scored.empty and qwen_scored is not None:
            st.subheader("🔍 Overlapping Sample Questions: Side-by-Side Answer Comparison")
            st.caption("Inspect identical test questions evaluated by both Qwen 2.5 (3B) and Google Gemini")

            for _, g_row in gem_scored.iterrows():
                q_id = g_row["id"]
                ds_label = g_row.get("source_dataset") or g_row.get("dataset", "Dataset")
                q_match_rows = qwen_scored[qwen_scored["id"] == q_id]
                q_row = q_match_rows.iloc[0] if not q_match_rows.empty else None

                em_g = "✅ Exact Match" if g_row.get("exact_match") == 1 else "❌ Mismatch"
                em_q = ("✅ Exact Match" if q_row.get("exact_match") == 1 else "❌ Mismatch") if q_row is not None else "—"

                with st.expander(f"{ds_label} — Question ID: {q_id} [Gemini: {em_g} | Qwen: {em_q}]", expanded=False):
                    st.markdown(f"**Question**: {g_row['question']}")
                    if g_row.get("context") and pd.notna(g_row["context"]):
                        st.caption(f"**Context**: {str(g_row['context'])[:250]}...")
                    st.markdown(f"**Ground Truth Reference**: `{g_row['reference_answer']}`")

                    col_ans1, col_ans2 = st.columns(2, gap="medium")
                    with col_ans1:
                        st.markdown("**🔴 Qwen 2.5 (3B) Answer**")
                        if q_row is not None:
                            st.info(f"\"{q_row['student_answer']}\"")
                            qm1, qm2 = st.columns(2)
                            qm1.metric("Exact Match", int(q_row.get("exact_match", 0)))
                            qm2.metric("ROUGE-L", f"{q_row.get('rouge_l', 0):.3f}")
                        else:
                            st.write("No matching answer in Qwen subset.")

                    with col_ans2:
                        st.markdown("**🟠 Google Gemini Answer**")
                        st.success(f"\"{g_row['gemini_answer']}\"")
                        gm1, gm2 = st.columns(2)
                        gm1.metric("Exact Match", int(g_row.get("exact_match", 0)))
                        gm2.metric("ROUGE-L", f"{g_row.get('rouge_l', 0):.3f}")


# ===========================================================================
# PAGE 2 — Visualizations Gallery
# ===========================================================================
elif page == "🖼️  Visualizations":

    st.markdown(
        """
        <div class='page-hero'>
            <h1>🖼️ Benchmark Visualizations Gallery</h1>
            <p>Comprehensive comparative performance charts generated in <code>figures/</code> · click to enlarge</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig_files = sorted(FIGURES_DIR.glob("*.png"))

    if not fig_files:
        st.info("No figures found in `figures/`. Run `python generate_visualizations.py` to produce them.", icon="ℹ️")
        st.stop()

    active_model = st.session_state.get("active_model", "Qwen 2.5 3B")

    # Strict canonical filename groups (6 Qwen, 6 Gemini, 6 Comparison)
    qwen_expected = ["qwen_bars.png", "qwen_radar.png", "qwen_boxplots.png", "qwen_heatmap.png", "qwen_latency.png", "qwen_correlations.png"]
    gemini_expected = ["gemini_bars.png", "gemini_radar.png", "gemini_boxplots.png", "gemini_heatmap.png", "gemini_latency.png", "gemini_correlations.png"]
    comparison_expected = ["comparison_bars.png", "comparison_scorecard.png", "comparison_f1.png", "comparison_latency.png", "comparison_radar.png", "comparison_heatmap.png"]

    cat_options = [
        "🔴 Qwen 2.5 (3B) Individual (6 Files)",
        "🟠 Google Gemini Individual (6 Files)",
        "⚔️ Head-to-Head Comparative (6 Files)",
        "🌐 All 18 Visualizations",
    ]

    # Map current active model to the default filter option
    if active_model == "Qwen 2.5 3B":
        default_idx = 0
    elif active_model == "Google Gemini":
        default_idx = 1
    elif active_model == "⚔️ Head-to-Head Comparison":
        default_idx = 2
    else:
        default_idx = 0

    # Filter controls
    f_c1, f_c2 = st.columns([2, 3])
    with f_c1:
        layout_mode = st.radio("Layout Mode", ["2-Column Grid", "Full Width"], horizontal=True)
    with f_c2:
        cat_filter = st.selectbox(
            "Filter Category",
            cat_options,
            index=default_idx,
            key=f"cat_filter_{active_model}",
        )

    # Strictly filter files based on the selected category
    if "Qwen" in cat_filter:
        fig_files = [FIGURES_DIR / name for name in qwen_expected if (FIGURES_DIR / name).exists()]
    elif "Gemini" in cat_filter:
        fig_files = [FIGURES_DIR / name for name in gemini_expected if (FIGURES_DIR / name).exists()]
    elif "Comparative" in cat_filter or "Comparison" in cat_filter:
        fig_files = [FIGURES_DIR / name for name in comparison_expected if (FIGURES_DIR / name).exists()]
    else:
        all_expected = qwen_expected + gemini_expected + comparison_expected
        fig_files = [FIGURES_DIR / name for name in all_expected if (FIGURES_DIR / name).exists()]

    st.caption(f"Displaying **{len(fig_files)}** publication-grade charts")
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    if layout_mode == "Full Width":
        for fp in fig_files:
            title = FIGURE_TITLES.get(fp.name, fp.stem.replace("_", " ").title())
            st.markdown(f"#### {title}")
            st.image(str(fp), use_container_width=True)
            st.caption(f"`{fp.name}` · {fp.stat().st_size / 1024:.1f} KB")
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    else:
        pairs = list(zip(fig_files[::2], fig_files[1::2]))
        if len(fig_files) % 2 == 1:
            pairs.append((fig_files[-1], None))

        for fl, fr in pairs:
            col_l, col_r = st.columns(2, gap="medium")
            with col_l:
                tl = FIGURE_TITLES.get(fl.name, fl.stem.replace("_", " ").title())
                st.markdown(f"**{tl}**")
                st.image(str(fl), use_container_width=True)
                st.caption(f"`{fl.name}` · {fl.stat().st_size / 1024:.1f} KB")

            if fr is not None:
                with col_r:
                    tr = FIGURE_TITLES.get(fr.name, fr.stem.replace("_", " ").title())
                    st.markdown(f"**{tr}**")
                    st.image(str(fr), use_container_width=True)
                    st.caption(f"`{fr.name}` · {fr.stat().st_size / 1024:.1f} KB")
            st.markdown("")


# ===========================================================================
# PAGE 3 — Live Model Playground (Supports Qwen & Gemini)
# ===========================================================================
elif page == "🧪  Live Model Playground":

    st.markdown(
        """
        <div class='page-hero'>
            <h1>🧪 Live Model Playground</h1>
            <p>Query local Ollama models (Qwen) or Google Gemini in real-time with latency measurement</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Try to import ollama ──
    try:
        import ollama as _ollama
        _ollama_available = True
    except ImportError:
        _ollama_available = False

    # ── Discover available models ──
    def get_ollama_models() -> list[str]:
        if not _ollama_available:
            return []
        try:
            resp = _ollama.list()
            return [m.model for m in resp.models] if resp.models else []
        except Exception:
            return []

    local_models = get_ollama_models()
    all_options = ["Qwen 2.5 (3B Local)", "Google Gemini (Flash)"]
    if local_models:
        all_options += [f"{m} (Ollama)" for m in local_models if m not in ("qwen2.5:3b", "qwen2.5:3b-instruct")]

    s1, s2, s3 = st.columns([2, 1, 1], gap="large")
    with s1:
        chosen_playground_model = st.selectbox("🤖 Choose Model", options=all_options)
    with s2:
        temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.0, 0.05)
    with s3:
        max_tokens = st.slider("📏 Max Output Tokens", 50, 500, 200, 25)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    subj_col, _ = st.columns([2, 3])
    with subj_col:
        subject = st.selectbox(
            "📚 Benchmark Domain",
            ["Science", "General Science", "Science Challenge", "Reading Comprehension", "General QA"],
        )

    question = st.text_area(
        "✏️ Enter Educational Question",
        height=120,
        placeholder="e.g. Which branch of biology studies animal behavior?\nWhen cold temperatures are produced in a chemical reaction, the reaction is known as...",
        key="pg_question_input",
    )

    run_col, clear_col, _ = st.columns([1, 1, 4])
    run_btn = run_col.button("▶  Run Query", use_container_width=True)
    clear_btn = clear_col.button("🗑  Clear", use_container_width=True)

    if clear_btn:
        for k in ["pg_ans", "pg_lat", "pg_mod"]:
            st.session_state.pop(k, None)
        st.rerun()

    if run_btn:
        q = question.strip()
        if not q:
            st.warning("Please enter a question to evaluate.", icon="⚠️")
        else:
            prompt = (
                f"You are an educational assessment assistant.\n"
                f"Subject: {subject}\n\n"
                f"Question: {q}\n\n"
                f"Answer in as few words as possible:\nAnswer:"
            )

            # ── Run Gemini ──
            if "Gemini" in chosen_playground_model:
                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    st.error("GEMINI_API_KEY or GOOGLE_API_KEY is not set in environment.")
                else:
                    with st.spinner("Calling Google Gemini via GenAI SDK …"):
                        t0 = time.perf_counter()
                        try:
                            from google import genai
                            from google.genai import types
                            client = genai.Client(api_key=api_key)
                            resp = client.models.generate_content(
                                model="gemini-3.8-flash",
                                contents=prompt,
                                config=types.GenerateContentConfig(
                                    temperature=temperature,
                                    max_output_tokens=max_tokens,
                                ),
                            )
                            el = time.perf_counter() - t0
                            st.session_state["pg_ans"] = resp.text.strip() if resp and resp.text else "No text returned"
                            st.session_state["pg_lat"] = el
                            st.session_state["pg_mod"] = "gemini-3.8-flash"
                        except Exception as exc:
                            st.session_state["pg_ans"] = f"ERROR: {exc}"
                            st.session_state["pg_lat"] = None
                            st.session_state["pg_mod"] = "Gemini"

            # ── Run Qwen / Ollama ──
            else:
                if not _ollama_available:
                    st.error("Ollama package not available.")
                else:
                    ollama_model = "qwen2.5:3b" if "Qwen" in chosen_playground_model else chosen_playground_model.replace(" (Ollama)", "")
                    with st.spinner(f"Querying {ollama_model} via Ollama …"):
                        t0 = time.perf_counter()
                        try:
                            res = _ollama.chat(
                                model=ollama_model,
                                messages=[{"role": "user", "content": prompt}],
                                options={"temperature": temperature, "num_predict": max_tokens},
                            )
                            el = time.perf_counter() - t0
                            st.session_state["pg_ans"] = res["message"]["content"].strip()
                            st.session_state["pg_lat"] = el
                            st.session_state["pg_mod"] = ollama_model
                        except Exception as exc:
                            st.session_state["pg_ans"] = f"ERROR: {exc}"
                            st.session_state["pg_lat"] = None
                            st.session_state["pg_mod"] = ollama_model

    # Display playground output
    if "pg_ans" in st.session_state:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        r_ans = st.session_state["pg_ans"]
        r_lat = st.session_state.get("pg_lat")
        r_mod = st.session_state.get("pg_mod", chosen_playground_model)

        h1, h2, h3 = st.columns(3)
        h1.metric("Model Used", r_mod)
        h2.metric("Latency", f"{r_lat:.2f}s" if r_lat is not None else "—")
        h3.metric("Temperature", f"{temperature:.2f}")

        if r_ans.startswith("ERROR:"):
            st.markdown(f"<div class='error-box'>{r_ans}</div>", unsafe_allow_html=True)
        else:
            st.markdown("**Generated Response**")
            st.markdown(f"<div class='response-box'>{r_ans}</div>", unsafe_allow_html=True)
