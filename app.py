"""
app.py - EduBench-Local Interactive Dashboard
==============================================
A Streamlit web dashboard for the EduBench-Local benchmarking pipeline.

Pages:
  1. Leaderboard & Analytics  - Metric cards, leaderboard table, per-subject
                                breakdown, score distributions and latency
                                charts drawn directly from CSV data.
  2. Visualizations           - Gallery of all PNG figures in figures/.
  3. Live Model Playground    - Query any local Ollama model in real-time,
                                see the answer and latency live.

Usage:
  streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path

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
# Global CSS — dark premium design
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Core background ── */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0d1117 0%, #161b2e 100%);
    }
    [data-testid="stSidebar"] {
        background: #0f1623;
        border-right: 1px solid #1e2a45;
    }

    /* ── Typography ── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        color: #e2e8f0;
    }
    h1 { color: #60a5fa; font-weight: 800; letter-spacing: -0.5px; }
    h2 { color: #93c5fd; font-weight: 700; }
    h3 { color: #bfdbfe; font-weight: 600; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #1e2d4a, #162040);
        border: 1px solid #2d4a7a;
        border-radius: 14px;
        padding: 18px 20px 14px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    [data-testid="metric-container"] label {
        color: #7aafff !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #1e2a45;
    }

    /* ── Sidebar nav ── */
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
        font-weight: 500;
        padding: 6px 0;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(37,99,235,0.35);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        box-shadow: 0 6px 20px rgba(59,130,246,0.5);
        transform: translateY(-1px);
    }

    /* ── Select / text inputs ── */
    .stSelectbox > div > div,
    .stTextArea > div > div,
    .stTextInput > div > div {
        background: #1a2540 !important;
        border: 1px solid #2d4a7a !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }

    /* ── Code / response box ── */
    .response-box {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px 24px;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #d1fae5;
        white-space: pre-wrap;
        max-height: 420px;
        overflow-y: auto;
    }
    .latency-badge {
        display: inline-block;
        background: linear-gradient(135deg, #065f46, #047857);
        color: #6ee7b7;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 10px;
        border: 1px solid #059669;
    }
    .error-box {
        background: #1f0a0a;
        border: 1px solid #7f1d1d;
        border-radius: 12px;
        padding: 16px 20px;
        color: #fca5a5;
        font-size: 0.9rem;
    }

    /* ── Section divider ── */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #2d4a7a, transparent);
        margin: 28px 0;
    }

    /* ── Page hero ── */
    .page-hero {
        background: linear-gradient(135deg, #0f2447 0%, #162040 100%);
        border: 1px solid #1e3a6e;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 28px;
    }
    .page-hero h1 { margin: 0 0 6px 0; font-size: 1.9rem; }
    .page-hero p  { color: #7aafff; margin: 0; font-size: 0.95rem; }

    /* ── Figure card ── */
    .fig-card {
        background: #111827;
        border: 1px solid #1e2a45;
        border-radius: 14px;
        padding: 16px;
        transition: box-shadow 0.2s;
    }
    .fig-card:hover {
        box-shadow: 0 0 0 2px #3b82f6, 0 8px 32px rgba(59,130,246,0.18);
    }

    /* ── Playground spinner override ── */
    .stSpinner > div { border-color: #3b82f6 transparent transparent transparent !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RESULTS_DIR = config.RESULTS_DIR
FIGURES_DIR = config.FIGURES_DIR

FIGURE_TITLES = {
    "01_overall_score_bars.png":     "Overall Score Comparison",
    "02_latency_vs_accuracy.png":    "Latency vs. Accuracy",
    "03_subject_heatmap.png":        "Subject Heatmap",
    "04_subject_bar_comparison.png": "Per-Subject Bar Comparison",
    "05_exact_match_rate.png":       "Exact Match Rate",
    "06_score_distribution.png":     "LLM Score Distribution",
    "07_latency_distribution.png":   "Latency Distribution",
    "08_tokens_vs_latency.png":      "Tokens vs. Latency",
}


@st.cache_data(ttl=30)
def load_leaderboard() -> pd.DataFrame | None:
    p = RESULTS_DIR / "leaderboard.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    num = ["n_questions", "exact_match_rate", "avg_llm_score_1_5",
           "avg_llm_score_0_10", "avg_latency_s", "n_errors"]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_scored() -> pd.DataFrame | None:
    p = RESULTS_DIR / "scored_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ["exact_match", "llm_score_0_10", "llm_score_1_5",
              "latency_s", "prompt_tokens", "completion_tokens", "total_tokens"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_subject_leaderboard() -> pd.DataFrame | None:
    p = RESULTS_DIR / "subject_leaderboard.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ["n_questions", "exact_match_rate", "avg_llm_score_1_5",
              "avg_llm_score_0_10", "avg_latency_s"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _score_color(val: float, lo: float, hi: float) -> str:
    """Map a value in [lo, hi] to a green-red gradient hex."""
    t = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    r = int((1 - t) * 220 + t * 34)
    g = int((1 - t) * 34  + t * 197)
    return f"#{r:02x}{g:02x}60"


def missing_data_warning(label: str) -> None:
    st.warning(
        f"**{label}** not found. "
        "Run the appropriate pipeline step to generate it.",
        icon="⚠️",
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 18px 0 12px;'>
            <span style='font-size:2.4rem;'>🎓</span><br>
            <span style='font-size:1.1rem; font-weight:800;
                         background: linear-gradient(90deg,#60a5fa,#818cf8);
                         -webkit-background-clip:text;
                         -webkit-text-fill-color:transparent;'>
                EduBench-Local
            </span><br>
            <span style='font-size:0.75rem; color:#4b6a9c;'>
                LLM Benchmarking Dashboard
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#1e2a45; margin:4px 0 16px;'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊  Leaderboard & Analytics", "🖼️  Visualizations", "🧪  Live Model Playground"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1e2a45; margin:16px 0 12px;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.72rem; color:#3a5278; text-align:center;'>"
        "Powered by Ollama · Streamlit<br>"
        f"Results dir: <code style='color:#4b6a9c'>{RESULTS_DIR.name}/</code>"
        "</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 1 — Leaderboard & Analytics
# ===========================================================================
if page == "📊  Leaderboard & Analytics":

    st.markdown(
        """
        <div class='page-hero'>
            <h1>📊 Leaderboard &amp; Analytics</h1>
            <p>Aggregated model performance across all evaluated subjects · auto-refreshes every 30 s</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lb = load_leaderboard()
    scored = load_scored()
    sub_lb = load_subject_leaderboard()

    if lb is None:
        missing_data_warning("results/leaderboard.csv")
        st.stop()

    # -- Top model spotlight -------------------------------------------------
    top = lb.iloc[0]

    # ── Metric cards ────────────────────────────────────────────────────────
    st.subheader("Top Model at a Glance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🏆 Top Model",  top["model"])
    c2.metric("🎯 LLM Score (1–5)", f"{top['avg_llm_score_1_5']:.3f}")
    c3.metric("✅ Exact Match",     f"{top['exact_match_rate']*100:.1f}%")
    c4.metric("⚡ Avg Latency",     f"{top['avg_latency_s']:.2f}s")
    c5.metric("📝 Questions",       int(top["n_questions"]))

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Full leaderboard table ───────────────────────────────────────────────
    st.subheader("Full Leaderboard")

    display_lb = lb.copy()
    display_lb["Rank"] = range(1, len(lb) + 1)
    display_lb["Exact Match %"]     = (display_lb["exact_match_rate"] * 100).map("{:.2f}%".format)
    display_lb["LLM Score (1–5)"]   = display_lb["avg_llm_score_1_5"].map("{:.4f}".format)
    display_lb["LLM Score (0–10)"]  = display_lb["avg_llm_score_0_10"].map("{:.4f}".format)
    display_lb["Avg Latency (s)"]   = display_lb["avg_latency_s"].map("{:.3f}".format)
    display_lb["Judge Errors"]      = display_lb["n_errors"].astype(int)
    display_lb["N Questions"]       = display_lb["n_questions"].astype(int)
    display_lb["Subjects"]          = display_lb["subjects"].str.replace("|", " · ", regex=False)

    show_cols = ["Rank", "model", "N Questions", "Exact Match %",
                 "LLM Score (1–5)", "LLM Score (0–10)", "Avg Latency (s)",
                 "Judge Errors", "Subjects"]
    st.dataframe(
        display_lb[show_cols].rename(columns={"model": "Model"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Per-subject breakdown ────────────────────────────────────────────────
    if sub_lb is not None and not sub_lb.empty:
        st.subheader("Per-Subject Breakdown")
        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
            disp_sub = sub_lb.copy()
            disp_sub["Exact Match %"]    = (disp_sub["exact_match_rate"] * 100).map("{:.2f}%".format)
            disp_sub["LLM Score (1–5)"]  = disp_sub["avg_llm_score_1_5"].map("{:.4f}".format)
            disp_sub["LLM Score (0–10)"] = disp_sub["avg_llm_score_0_10"].map("{:.4f}".format)
            disp_sub["Avg Latency (s)"]  = disp_sub["avg_latency_s"].map("{:.3f}".format)
            disp_sub["N"]                = disp_sub["n_questions"].astype(int)
            st.dataframe(
                disp_sub[["model", "subject", "N", "Exact Match %",
                           "LLM Score (1–5)", "LLM Score (0–10)", "Avg Latency (s)"]
                         ].rename(columns={"model": "Model", "subject": "Subject"}),
                use_container_width=True,
                hide_index=True,
            )

        with col_right:
            # Mini bar chart — LLM score by subject
            pivot = sub_lb.pivot_table(
                index="subject", columns="model",
                values="avg_llm_score_1_5", aggfunc="mean"
            ).reset_index()
            chart_df = pivot.set_index("subject")
            st.markdown("**LLM Score (1–5) by Subject**")
            st.bar_chart(chart_df, height=280)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Scored results deep-dive ─────────────────────────────────────────────
    if scored is not None and not scored.empty:
        st.subheader("Detailed Results Explorer")

        # Filters row
        f1, f2, f3 = st.columns([2, 2, 2])
        with f1:
            models_avail = sorted(scored["model"].unique())
            sel_model = st.selectbox("Filter by Model", ["All"] + models_avail, key="lb_model")
        with f2:
            subjects_avail = sorted(scored["subject"].unique())
            sel_subj = st.selectbox("Filter by Subject", ["All"] + subjects_avail, key="lb_subj")
        with f3:
            score_range = st.slider(
                "LLM Score (1–5) range", 1, 5, (1, 5), key="lb_score"
            )

        filtered = scored.copy()
        if sel_model != "All":
            filtered = filtered[filtered["model"] == sel_model]
        if sel_subj != "All":
            filtered = filtered[filtered["subject"] == sel_subj]
        filtered = filtered[
            filtered["llm_score_1_5"].between(score_range[0], score_range[1])
        ]

        st.caption(f"Showing **{len(filtered):,}** of **{len(scored):,}** rows")

        show_scored_cols = ["model", "subject", "id", "question",
                            "reference_answer", "student_answer",
                            "exact_match", "llm_score_1_5", "latency_s"]
        available = [c for c in show_scored_cols if c in filtered.columns]
        st.dataframe(
            filtered[available].rename(columns={
                "model": "Model", "subject": "Subject", "id": "ID",
                "question": "Question", "reference_answer": "Ref. Answer",
                "student_answer": "Student Answer", "exact_match": "EM",
                "llm_score_1_5": "LLM(1–5)", "latency_s": "Latency(s)",
            }),
            use_container_width=True,
            hide_index=True,
            height=360,
        )

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        # ── Score & latency distributions ───────────────────────────────────
        st.subheader("Score & Latency Distributions")
        dist_l, dist_r = st.columns(2, gap="large")

        with dist_l:
            st.markdown("**LLM Score (1–5) distribution**")
            score_hist = (
                scored["llm_score_1_5"]
                .value_counts()
                .sort_index()
                .rename_axis("Score")
                .reset_index(name="Count")
            )
            st.bar_chart(score_hist.set_index("Score"), height=260)

        with dist_r:
            st.markdown("**Latency distribution (s) — binned**")
            lat_hist = (
                pd.cut(scored["latency_s"].dropna(), bins=20)
                .value_counts()
                .sort_index()
            )
            lat_df = pd.DataFrame({
                "Bucket": lat_hist.index.astype(str),
                "Count":  lat_hist.values,
            })
            st.bar_chart(lat_df.set_index("Bucket"), height=260)


# ===========================================================================
# PAGE 2 — Visualizations
# ===========================================================================
elif page == "🖼️  Visualizations":

    st.markdown(
        """
        <div class='page-hero'>
            <h1>🖼️ Visualizations</h1>
            <p>All performance charts generated by <code>step4_visualize.py</code> · click any image to expand</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scan figures dir
    fig_files = sorted(FIGURES_DIR.glob("*.png"))

    if not fig_files:
        st.info(
            "No figures found in `figures/`. Run `python step4_visualize.py` to generate them.",
            icon="ℹ️",
        )
        st.stop()

    st.caption(f"Found **{len(fig_files)}** charts in `{FIGURES_DIR.as_posix()}`")

    # ── Gallery toggle ───────────────────────────────────────────────────────
    view_mode = st.radio(
        "Layout", ["2-column grid", "Full width"],
        horizontal=True, label_visibility="collapsed"
    )

    if view_mode == "Full width":
        for fig_path in fig_files:
            title = FIGURE_TITLES.get(fig_path.name, fig_path.stem.replace("_", " ").title())
            st.markdown(f"#### {title}")
            st.image(str(fig_path), use_container_width=True)
            size_kb = fig_path.stat().st_size / 1024
            st.caption(f"`{fig_path.name}` · {size_kb:.1f} KB")
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    else:
        # 2-column grid
        pairs = list(zip(fig_files[::2], fig_files[1::2]))
        if len(fig_files) % 2 == 1:
            pairs.append((fig_files[-1], None))

        for left_path, right_path in pairs:
            col_l, col_r = st.columns(2, gap="medium")

            with col_l:
                title_l = FIGURE_TITLES.get(left_path.name, left_path.stem.replace("_", " ").title())
                st.markdown(f"**{title_l}**")
                st.image(str(left_path), use_container_width=True)
                sz = left_path.stat().st_size / 1024
                st.caption(f"`{left_path.name}` · {sz:.1f} KB")

            if right_path is not None:
                with col_r:
                    title_r = FIGURE_TITLES.get(right_path.name, right_path.stem.replace("_", " ").title())
                    st.markdown(f"**{title_r}**")
                    st.image(str(right_path), use_container_width=True)
                    sz = right_path.stat().st_size / 1024
                    st.caption(f"`{right_path.name}` · {sz:.1f} KB")

            st.markdown("")


# ===========================================================================
# PAGE 3 — Live Model Playground
# ===========================================================================
elif page == "🧪  Live Model Playground":

    st.markdown(
        """
        <div class='page-hero'>
            <h1>🧪 Live Model Playground</h1>
            <p>Query any local Ollama model in real-time · results and latency shown instantly</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Try to import ollama ─────────────────────────────────────────────────
    try:
        import ollama as _ollama
        _ollama_available = True
    except ImportError:
        _ollama_available = False

    if not _ollama_available:
        st.error("The `ollama` Python package is not installed in this environment.\n\n"
                 "```\npip install ollama\n```")
        st.stop()

    # ── Discover available local models ──────────────────────────────────────
    @st.cache_data(ttl=15)
    def get_local_models() -> list[str]:
        try:
            resp = _ollama.list()
            return [m.model for m in resp.models] if resp.models else []
        except Exception:
            return []

    local_models = get_local_models()

    # Always surface the pipeline models prominently
    pipeline_models = list(dict.fromkeys(
        config.MODELS + [config.JUDGE_MODEL] + local_models
    ))

    # ── Settings row ─────────────────────────────────────────────────────────
    s1, s2, s3 = st.columns([2, 1, 1], gap="large")

    with s1:
        if local_models:
            model_choice = st.selectbox(
                "🤖 Select Model",
                options=pipeline_models,
                help="Models installed locally via Ollama",
            )
        else:
            model_choice = st.selectbox(
                "🤖 Select Model (Ollama not reachable — choose manually)",
                options=pipeline_models,
            )

    with s2:
        temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.2, 0.05)

    with s3:
        max_tokens = st.slider("📏 Max Tokens", 50, 500, 200, 25)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Subject & question ────────────────────────────────────────────────────
    subj_col, _ = st.columns([2, 3])
    with subj_col:
        subject = st.selectbox(
            "📚 Subject (optional context)",
            ["General", "Science", "Reading Comprehension", "Mathematics",
             "History", "Literature", "Geography", "Other"],
        )

    question = st.text_area(
        "✏️ Enter your educational question",
        height=120,
        placeholder="e.g.  What is the powerhouse of the cell?\n"
                    "      Explain Newton's second law of motion.\n"
                    "      What caused the French Revolution?",
        key="playground_question",
    )

    # Example prompts
    with st.expander("💡 Try an example question", expanded=False):
        examples = [
            "What is the process by which plants make their own food using sunlight?",
            "Explain the difference between mitosis and meiosis.",
            "What is the significance of the Magna Carta in the development of democracy?",
            "Describe how a transistor works in simple terms.",
            "What is the central limit theorem and why is it important in statistics?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:30]}"):
                st.session_state["playground_question"] = ex
                st.rerun()

    # ── Submit ────────────────────────────────────────────────────────────────
    run_col, clear_col, _ = st.columns([1, 1, 4])
    run_btn   = run_col.button("▶  Run Query", use_container_width=True)
    clear_btn = clear_col.button("🗑  Clear", use_container_width=True)

    if clear_btn:
        for k in ["pg_response", "pg_latency", "pg_model", "pg_tokens"]:
            st.session_state.pop(k, None)
        st.rerun()

    # ── Execute ───────────────────────────────────────────────────────────────
    if run_btn:
        q = question.strip()
        if not q:
            st.warning("Please enter a question before running.", icon="⚠️")
        else:
            prompt = (
                f"You are a knowledgeable educational assistant.\n"
                f"Subject: {subject}\n\n"
                f"Question: {q}\n\n"
                f"Answer clearly and accurately:"
            )

            with st.spinner(f"Querying **{model_choice}** via Ollama …"):
                t0 = time.perf_counter()
                try:
                    resp = _ollama.chat(
                        model=model_choice,
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": temperature, "num_predict": max_tokens},
                    )
                    elapsed = time.perf_counter() - t0
                    answer  = resp["message"]["content"].strip()
                    usage   = resp.get("usage", {})
                    total_toks = (
                        usage.get("total_tokens")
                        or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
                        or resp.get("eval_count", "—")
                    )
                    st.session_state["pg_response"] = answer
                    st.session_state["pg_latency"]  = elapsed
                    st.session_state["pg_model"]    = model_choice
                    st.session_state["pg_tokens"]   = total_toks

                except Exception as exc:
                    st.session_state["pg_response"] = f"ERROR: {exc}"
                    st.session_state["pg_latency"]  = None
                    st.session_state["pg_model"]    = model_choice
                    st.session_state["pg_tokens"]   = None

    # ── Display result ─────────────────────────────────────────────────────────
    if "pg_response" in st.session_state:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        resp_text  = st.session_state["pg_response"]
        lat        = st.session_state.get("pg_latency")
        resp_model = st.session_state.get("pg_model", model_choice)
        n_toks     = st.session_state.get("pg_tokens", "—")

        # Header row
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Model",    resp_model)
        h2.metric("Latency",  f"{lat:.2f}s"   if lat is not None else "—")
        h3.metric("Tokens",   str(n_toks)      if n_toks else "—")
        h4.metric("Temp.",    f"{temperature:.2f}")

        st.markdown("")

        if resp_text.startswith("ERROR:"):
            st.markdown(
                f"<div class='error-box'>{resp_text}</div>",
                unsafe_allow_html=True,
            )
            st.info(
                "Make sure Ollama is running (`ollama serve`) and the model is "
                f"pulled (`ollama pull {resp_model}`).",
                icon="ℹ️",
            )
        else:
            st.markdown("**Model Response**")
            st.markdown(
                f"<div class='response-box'>{resp_text}</div>",
                unsafe_allow_html=True,
            )

            # ── Compare against leaderboard if same model ────────────────────
            lb = load_leaderboard()
            if lb is not None:
                match = lb[lb["model"] == resp_model]
                if not match.empty:
                    bm = match.iloc[0]
                    st.markdown("")
                    st.markdown("**Benchmark Context** — how this model performed in EduBench-Local")
                    bm1, bm2, bm3, bm4 = st.columns(4)
                    bm1.metric("Benchmarked LLM Score (1–5)", f"{bm['avg_llm_score_1_5']:.3f}")
                    bm2.metric("Exact Match Rate",            f"{bm['exact_match_rate']*100:.1f}%")
                    bm3.metric("Bench Avg Latency",           f"{bm['avg_latency_s']:.2f}s")
                    bm4.metric("Questions Evaluated",         int(bm["n_questions"]))

                    # Flag if this query's latency is anomalous
                    if lat is not None:
                        delta = lat - bm["avg_latency_s"]
                        if delta > 2.0:
                            st.warning(
                                f"This query took **{lat:.2f}s**, which is "
                                f"**{delta:+.2f}s** above the benchmark average "
                                f"({bm['avg_latency_s']:.2f}s). "
                                "Ollama may be under load.",
                                icon="⚠️",
                            )
