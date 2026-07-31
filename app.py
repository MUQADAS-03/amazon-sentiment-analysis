"""
Amazon Customer Reviews — Sentiment Intelligence Dashboard

Run with: streamlit run app.py
"""

import re
import json
import os

import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Amazon | Sentiment Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ----------------------------------------------------------------------------
# Design tokens — pulled straight from Amazon's own storefront palette
# ----------------------------------------------------------------------------
NAVY_DARK   = "#131921"   # Amazon top-nav black
NAVY        = "#232F3E"   # Amazon secondary nav
ORANGE      = "#FF9900"   # Amazon CTA orange
GOLD        = "#FEBD69"   # Amazon secondary accent
STAR_GOLD   = "#FFA41C"   # Amazon star-rating gold
BG          = "#EAEDED"   # Amazon page background grey
CARD_BG     = "#FFFFFF"
TEXT_DARK   = "#0F1111"   # Amazon body text
TEXT_MUTED  = "#565959"   # Amazon secondary text
POSITIVE    = "#067D62"   # Amazon "in stock" green
NEGATIVE    = "#B12704"   # Amazon price-red
LINE        = "#D5D9D9"

# ----------------------------------------------------------------------------
# Global styling
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{ background-color: {BG}; }}

    /* ---- Hide Streamlit's default white top header bar ---- */
    header[data-testid="stHeader"] {{
        display: none !important;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {NAVY_DARK};
    }}
    section[data-testid="stSidebar"] * {{
        color: #F2F2F2 !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: #3A4553;
    }}

    /* ---- container layout adjustments ---- */
    .block-container {{
        padding-top: 0rem !important;
        padding-bottom: 3rem !important;
        max-width: 100% !important;
    }}

    /* ---- top storefront bar fixed layout ---- */
    .amz-topbar {{
        background: linear-gradient(180deg, {NAVY_DARK} 0%, {NAVY} 100%);
        margin: 0rem -5rem 1.5rem -5rem;
        padding: 22px 40px 18px 40px;
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        flex-wrap: wrap;
        border-bottom: 3px solid {ORANGE};
    }}
    .amz-wordmark {{
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 2.1rem;
        color: #FFFFFF;
        letter-spacing: -0.5px;
        line-height: 1;
    }}
    .amz-task {{
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        color: {GOLD};
        text-align: right;
    }}
    .amz-task-sub {{
        font-size: 0.8rem;
        color: #B7C0CC;
        text-align: right;
        margin-top: 2px;
    }}

    .app-subtitle {{
        font-size: 1.05rem;
        color: {TEXT_MUTED};
        margin-top: -8px;
        margin-bottom: 6px;
    }}
    .section-header {{
        font-family: 'Poppins', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: {NAVY_DARK};
        border-bottom: 3px solid {ORANGE};
        display: inline-block;
        padding-bottom: 6px;
        margin-top: 24px;
        margin-bottom: 16px;
    }}

    /* ---- stat / info cards ---- */
    .stat-card {{
        background-color: {CARD_BG};
        border: 1px solid {LINE};
        border-left: 6px solid {ORANGE};
        border-radius: 8px;
        padding: 16px 20px;
        height: 100%;
    }}
    .stat-label {{
        font-size: 0.82rem;
        font-weight: 600;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }}
    .stat-value {{
        font-family: 'Poppins', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: {NAVY_DARK};
        margin-top: 2px;
    }}
    .stat-delta-pos {{ color: {POSITIVE}; font-weight: 600; font-size: 0.9rem; }}
    .stat-delta-neg {{ color: {NEGATIVE}; font-weight: 600; font-size: 0.9rem; }}

    .info-box {{
        background-color: {CARD_BG};
        border: 1px solid {LINE};
        border-radius: 8px;
        padding: 18px 22px;
        color: {TEXT_DARK};
        line-height: 1.55;
    }}
    .info-box b {{ color: {NAVY_DARK}; }}

    .how-card {{
        background-color: {CARD_BG};
        border: 1px solid {LINE};
        border-top: 4px solid {GOLD};
        border-radius: 8px;
        padding: 18px 20px;
        height: 100%;
    }}
    .how-card .idx {{
        display: inline-block;
        background-color: {NAVY_DARK};
        color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 0.85rem;
        border-radius: 50%;
        width: 26px; height: 26px;
        text-align: center;
        line-height: 26px;
        margin-bottom: 8px;
    }}

    /* ---- prediction result ---- */
    .verdict-card {{
        border-radius: 10px;
        padding: 20px 26px;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 14px;
    }}
    .verdict-label {{
        font-family: 'Poppins', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
    }}
    .verdict-sub {{
        font-size: 0.85rem;
        opacity: 0.9;
        margin-top: 2px;
    }}
    .verified-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.5);
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }}

    div[data-testid="stButton"] button {{
        background-color: {ORANGE};
        color: {NAVY_DARK};
        font-weight: 700;
        border: 1px solid #A66A00;
        border-radius: 8px;
    }}
    div[data-testid="stButton"] button:hover {{
        background-color: {GOLD};
        color: {NAVY_DARK};
        border: 1px solid #A66A00;
    }}

    /* ---- full-page Amazon frame ---- */
    div[data-testid="stAppViewContainer"] > .main {{
        border-left: 6px solid {ORANGE};
        border-right: 6px solid {ORANGE};
    }}

    /* ---- hero banner with box-tape watermark ---- */
    .amz-hero {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, {NAVY_DARK} 0%, {NAVY} 55%, {NAVY_DARK} 100%);
        border-radius: 12px;
        border: 1px solid {ORANGE};
        padding: 26px 30px;
        margin-bottom: 22px;
    }}
    .amz-hero::before {{
        content: "";
        position: absolute;
        top: -40px; right: -60px;
        width: 260px; height: 260px;
        background: repeating-linear-gradient(45deg, {ORANGE}22 0 14px, transparent 14px 28px);
        border-radius: 50%;
        pointer-events: none;
    }}
    .amz-hero-title {{
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.4rem;
        color: #FFFFFF;
        position: relative;
        z-index: 1;
    }}
    .amz-hero-sub {{
        color: #C9D1D9;
        font-size: 0.95rem;
        max-width: 640px;
        position: relative;
        z-index: 1;
        line-height: 1.5;
        margin-top: 6px;
    }}

    /* ---- Prime-style ribbon badge for KPI cards ---- */
    .prime-ribbon {{
        display: inline-block;
        background-color: {NAVY_DARK};
        color: {GOLD};
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 2px 9px;
        border-radius: 20px;
        margin-bottom: 6px;
    }}

    /* ---- insight / trend cards ---- */
    .trend-card {{
        background-color: {CARD_BG};
        border: 1px solid {LINE};
        border-top: 5px solid {ORANGE};
        border-radius: 10px;
        padding: 18px 20px;
        height: 100%;
    }}
    .trend-card-title {{
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        color: {NAVY_DARK};
        margin-bottom: 10px;
    }}
    .variation-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid {LINE};
        font-size: 0.88rem;
    }}
    .variation-row:last-child {{ border-bottom: none; }}
    .variation-name {{ color: {TEXT_DARK}; font-weight: 600; }}
    .variation-meta {{ color: {TEXT_MUTED}; font-size: 0.8rem; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Signature element — Amazon wordmark + smile-arrow, reused on every page
# ----------------------------------------------------------------------------
def render_topbar(task_title: str, task_sub: str):
    st.markdown(f"""
    <div class="amz-topbar">
        <div>
            <span class="amz-wordmark">amazon</span>
            <svg width="120" height="22" viewBox="0 0 120 22" style="display:block; margin-top:-2px;">
                <path d="M4 4 C 30 22, 90 22, 116 4" stroke="{ORANGE}" stroke-width="3"
                      fill="none" stroke-linecap="round"/>
                <path d="M104 2 L118 4 L108 14" stroke="{ORANGE}" stroke-width="3"
                      fill="none" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <div>
            <div class="amz-task">{task_title}</div>
            <div class="amz-task-sub">{task_sub}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_stars(confidence: float) -> str:
    """Render an Amazon-style star rating (0-5 scale, half-star precision)."""
    rating = round(confidence * 5 * 2) / 2
    stars_html = []
    for i in range(1, 6):
        if rating >= i:
            fill = "100%"
        elif rating >= i - 0.5:
            fill = "50%"
        else:
            fill = "0%"
        gid = f"star-grad-{i}-{int(confidence * 1000)}"
        stars_html.append(f"""
        <svg width="22" height="22" viewBox="0 0 24 24">
            <defs>
                <linearGradient id="{gid}">
                    <stop offset="{fill}" stop-color="{STAR_GOLD}"/>
                    <stop offset="{fill}" stop-color="#E3E6E6"/>
                </linearGradient>
            </defs>
            <path fill="url(#{gid})" stroke="{STAR_GOLD}" stroke-width="0.5"
                  d="M12 2.5l2.9 6.1 6.6.7-4.9 4.5 1.3 6.6L12 17l-5.9 3.4 1.3-6.6-4.9-4.5 6.6-.7z"/>
        </svg>""")
    return f'<div style="display:flex; gap:2px;">{"".join(stars_html)}</div>'


# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))
    with open(os.path.join(MODELS_DIR, "model_name.txt")) as f:
        model_name = f.read().strip()
    return model, vectorizer, model_name


@st.cache_data
def load_stats():
    with open(os.path.join(MODELS_DIR, "dataset_stats.json")) as f:
        stats = json.load(f)
    comparison = pd.read_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index_col=0)
    return stats, comparison


@st.cache_data
def load_business_insights():
    """
    Derives extra KPIs/trends from the raw reviews file for the Home page.
    Returns None if the raw CSV isn't found so the dashboard degrades gracefully.
    """
    csv_path = os.path.join(DATA_DIR, "amazon_reviews.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        raw = pd.read_csv(csv_path)
        raw = raw.dropna(subset=["verified_reviews"]).drop_duplicates(subset=["verified_reviews"])
        raw["date"] = pd.to_datetime(raw["date"], format="%d-%b-%y", errors="coerce")
        raw = raw.dropna(subset=["date"])

        avg_rating = float(raw["rating"].mean())
        rating_counts = raw["rating"].value_counts().sort_index()

        monthly = raw.set_index("date").resample("MS").agg(
            volume=("feedback", "size"),
            avg_sentiment=("feedback", "mean"),
        )
        monthly.index = monthly.index.strftime("%b %Y")

        top_variations = (
            raw.groupby("variation")
            .agg(reviews=("feedback", "size"), positive_rate=("feedback", "mean"), avg_rating=("rating", "mean"))
            .sort_values("reviews", ascending=False)
            .head(5)
        )

        return {
            "avg_rating": avg_rating,
            "rating_counts": rating_counts,
            "monthly": monthly,
            "top_variations": top_variations,
        }
    except Exception:
        return None


STOP_WORDS = None
def get_stopwords():
    global STOP_WORDS
    if STOP_WORDS is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        STOP_WORDS = set(ENGLISH_STOP_WORDS)
    return STOP_WORDS


def normalize(word: str) -> str:
    for suffix in ("ing", "edly", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[: -len(suffix)]
    return word


def clean_text(text: str) -> str:
    """Identical cleaning pipeline used in the Part 1 notebook."""
    stop_words = get_stopwords()
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [normalize(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


def stat_card(label, value, delta=None, delta_positive=True, ribbon=None):
    delta_html = ""
    if delta is not None:
        cls = "stat-delta-pos" if delta_positive else "stat-delta-neg"
        delta_html = f'<div class="{cls}">{delta}</div>'
    ribbon_html = f'<div class="prime-ribbon">{ribbon}</div>' if ribbon else ""
    st.markdown(f"""
    <div class="stat-card">
        {ribbon_html}
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.markdown("## Sentiment Intelligence")
st.sidebar.caption("Customer Review Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Data Overview", "Sentiment Predictor"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("Dataset: Amazon customer reviews\nModel: TF-IDF + classical ML classifier")

# ============================================================================
# PAGE 1 — HOME
# ============================================================================
if page == "Home":
    render_topbar("Voice of the Customer", "Sentiment Intelligence Dashboard")
    st.write("")

    stats, comparison = load_stats()
    insights = load_business_insights()

    st.markdown(f"""
    <div class="amz-hero">
        <div class="amz-hero-title">📦 Turning raw customer reviews into product decisions</div>
        <div class="amz-hero-sub">
        {stats['total_reviews']:,} verified reviews were cleaned, vectorized with TF-IDF, and used to
        train and benchmark multiple classifiers. <b style="color:{GOLD};">{stats['best_model']}</b>
        won on F1-score and now powers instant, live sentiment scoring on the Predictor page.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stat_card("Total Reviews Analyzed", f"{stats['total_reviews']:,}", ribbon="Dataset")
    with col2:
        stat_card("Positive Reviews", f"{stats['positive_reviews']:,}",
                   f"▲ {stats['positive_reviews']/stats['total_reviews']*100:.1f}%", True, ribbon="Sentiment")
    with col3:
        stat_card("Negative Reviews", f"{stats['negative_reviews']:,}",
                   f"▼ {stats['negative_reviews']/stats['total_reviews']*100:.1f}%", False, ribbon="Sentiment")
    with col4:
        stat_card("Best Model F1-Score", f"{stats['best_f1']*100:.1f}%", ribbon="Model")

    with st.expander("What is sentiment analysis?"):
        st.markdown("""
        An NLP technique that determines whether text expresses a positive or negative opinion.
        Text is cleaned and converted into numerical features, and the model learns which word
        patterns correlate with each class — used to monitor reviews, social mentions, and
        support tickets at a scale no human team could read manually.
        """)

    # ------------------------------------------------------------------
    # Business Insights & Trends
    # ------------------------------------------------------------------
    st.markdown('<div class="section-header">Business Insights &amp; Trends</div>', unsafe_allow_html=True)

    if insights is None:
        st.info("Drop `amazon_reviews.csv` into the `data/` folder to unlock rating trends, "
                "monthly volume, and top product-variation breakdowns here.")
    else:
        k1, k2, k3 = st.columns(3)
        with k1:
            stat_card("Average Star Rating", f"{insights['avg_rating']:.2f} / 5", ribbon="Customer Voice")
        with k2:
            best_month = insights["monthly"]["volume"].idxmax()
            stat_card("Busiest Review Month", best_month,
                      f"{int(insights['monthly']['volume'].max()):,} reviews", True, ribbon="Volume")
        with k3:
            top_var = insights["top_variations"].index[0]
            top_rate = insights["top_variations"].iloc[0]["positive_rate"] * 100
            stat_card("Top Variation by Volume", top_var, f"{top_rate:.0f}% positive", True, ribbon="Product")

        t1, t2 = st.columns([1.3, 1])
        with t1:
            st.markdown('<div class="trend-card"><div class="trend-card-title">📈 Monthly Review Volume &amp; Sentiment</div>', unsafe_allow_html=True)
            st.line_chart(insights["monthly"][["volume"]], color=[ORANGE], height=240)
            st.caption("Review volume by month — spikes flag launches, promotions, or PR moments worth digging into.")
            st.markdown("</div>", unsafe_allow_html=True)
        with t2:
            st.markdown('<div class="trend-card"><div class="trend-card-title">⭐ Rating Distribution</div>', unsafe_allow_html=True)
            rc = insights["rating_counts"].rename("Reviews")
            st.bar_chart(rc, color=[GOLD], height=240)
            st.caption("1–5 star breakdown across all verified reviews.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="trend-card" style="margin-top:16px;"><div class="trend-card-title">🏆 Top Product Variations</div>', unsafe_allow_html=True)
        rows_html = ""
        for name, row in insights["top_variations"].iterrows():
            rows_html += f"""
            <div class="variation-row">
                <span class="variation-name">{name}</span>
                <span class="variation-meta">{int(row['reviews']):,} reviews &nbsp;·&nbsp;
                {row['avg_rating']:.1f}★ &nbsp;·&nbsp;
                <b style="color:{POSITIVE};">{row['positive_rate']*100:.0f}% positive</b></span>
            </div>"""
        st.markdown(rows_html + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">How to Use This Dashboard</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="how-card"><span class="idx">1</span><br>
        <b style="color:{NAVY_DARK}">Data Overview</b><br>
        Explore class distribution and word clouds derived from the training data.</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="how-card"><span class="idx">2</span><br>
        <b style="color:{NAVY_DARK}">Sentiment Predictor</b><br>
        Type or paste any review text and get an instant sentiment prediction.</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="how-card"><span class="idx">3</span><br>
        <b style="color:{NAVY_DARK}">Confidence Score</b><br>
        Every prediction is returned as a star rating, not just a label.</div>""", unsafe_allow_html=True)

# ============================================================================
# PAGE 2 — DATA OVERVIEW
# ============================================================================
elif page == "Data Overview":
    render_topbar("Data Overview", "Class Balance & Model Comparison")
    st.markdown('<div class="app-subtitle">Class balance, vocabulary patterns, and model comparison from Part 1</div>', unsafe_allow_html=True)
    st.write("")

    stats, comparison = load_stats()

    st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.3])
    with col1:
        dist_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative"],
            "Count": [stats["positive_reviews"], stats["negative_reviews"]],
        })
        st.dataframe(dist_df, hide_index=True, width="stretch")
        st.markdown(f"""
        <div class="info-box">
        The dataset is imbalanced — roughly
        <b>{stats['positive_reviews']/stats['negative_reviews']:.1f} : 1</b> positive-to-negative
        ratio. This was handled during training with stratified splitting and class-weighted
        models so the classifier does not simply default to predicting "Positive".
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image(os.path.join(ASSETS_DIR, "outputs_class_distribution.png"), width="stretch")

    st.markdown('<div class="section-header">Word Clouds — Positive vs Negative Reviews</div>', unsafe_allow_html=True)
    st.image(os.path.join(ASSETS_DIR, "outputs_wordclouds.png"), width="stretch")
    st.caption("Positive reviews are dominated by usability and satisfaction terms; negative "
               "reviews surface connectivity and reliability complaints.")

    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image(os.path.join(ASSETS_DIR, "outputs_model_comparison.png"), width="stretch")
    with col2:
        st.dataframe(comparison.style.highlight_max(axis=0, color=f"{POSITIVE}33"), width="stretch")
        st.markdown(f"""
        <div class="info-box">
        <b>{stats['best_model']}</b> was selected as the production model, deployed on the
        Sentiment Predictor page, with an F1-score of <b>{stats['best_f1']*100:.1f}%</b> and
        accuracy of <b>{stats['best_accuracy']*100:.1f}%</b>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    st.image(os.path.join(ASSETS_DIR, "outputs_confusion_matrices.png"), width="stretch")

# ============================================================================
# PAGE 3 — SENTIMENT PREDICTOR
# ============================================================================
elif page == "Sentiment Predictor":
    render_topbar("Sentiment Predictor", "Instant Review Classification")
    st.markdown('<div class="app-subtitle">Type or paste any product review to classify its sentiment</div>', unsafe_allow_html=True)
    st.write("")

    model, vectorizer, model_name = load_model_and_vectorizer()
    st.caption(f"Serving model: {model_name}")

    review_text = st.text_area(
        "Review text",
        height=140,
        placeholder="e.g. The sound quality is amazing and it connects instantly every time...",
        label_visibility="collapsed",
    )

    predict_clicked = st.button("Predict Sentiment", type="primary", use_container_width=False)

    if predict_clicked:
        if not review_text.strip():
            st.warning("Please enter some review text before predicting.")
        else:
            cleaned = clean_text(review_text)
            if not cleaned:
                st.warning("The text contained no usable words after cleaning. Try a more descriptive review.")
            else:
                vec = vectorizer.transform([cleaned])
                pred = model.predict(vec)[0]
                proba = model.predict_proba(vec)[0]
                classes = list(model.classes_)
                pos_idx = classes.index(1)
                neg_idx = classes.index(0)
                pos_conf = proba[pos_idx]
                neg_conf = proba[neg_idx]
                confidence = pos_conf if pred == 1 else neg_conf

                label = "Positive" if pred == 1 else "Negative"
                color = POSITIVE if pred == 1 else NEGATIVE

                st.markdown(f"""
                <div class="verdict-card" style="background-color:{color};">
                    <div>
                        <div class="verdict-label">{label} Sentiment</div>
                        <div class="verdict-sub">Confidence score: {confidence*100:.1f}%</div>
                    </div>
                    <div style="display:flex; align-items:center; gap:16px;">
                        {render_stars(confidence)}
                        <span class="verified-badge">
                            <svg width="14" height="14" viewBox="0 0 24 24">
                                <path fill="#FFFFFF" d="M12 2l2.4 4.9 5.4.8-3.9 3.8.9 5.4L12 14.4 7.2 16.9l.9-5.4-3.9-3.8 5.4-.8z"/>
                            </svg>
                            Verified Analysis
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.write("")
                st.markdown('<div class="section-header">Prediction Breakdown</div>', unsafe_allow_html=True)
                breakdown = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative"],
                    "Probability": [pos_conf, neg_conf],
                })
                st.bar_chart(breakdown.set_index("Sentiment"), color=[ORANGE], width="stretch")

                with st.expander("View cleaned text used by the model"):
                    st.code(cleaned, language=None)

    st.markdown('<div class="section-header">About the Model</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
    Input text is cleaned using the exact same pipeline as Part 1 of the notebook — lowercasing,
    URL/HTML/punctuation removal, stopword removal, and suffix normalization — before being
    converted into TF-IDF features and passed to the trained <b>{model_name}</b> classifier.
    This ensures the prediction here is consistent with the evaluation reported on the
    Data Overview page.
    </div>
    """, unsafe_allow_html=True)