import re
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------
#  Page Config
# -------------------------------------------------------
st.set_page_config
    page_title="Fake News Detector",
    page_icon="newspaper",
    layout="centered",
)

# -------------------------------------------------------
#  CSS
# -------------------------------------------------------
st.markdown("""
<style>
.main { background: #f0f4f8; }

.app-header { text-align: center; padding: 2rem 0 1.2rem; }
.app-header h1 { font-size: 2.5rem; font-weight: 800; color: #0f172a; margin: 0; letter-spacing: -1px; }
.app-header p  { color: #64748b; font-size: 1rem; margin-top: 6px; }

.card {
    background: #ffffff;
    border-radius: 18px;
    padding: 28px 32px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.07);
    margin-bottom: 20px;
}

.sec-label {
    font-size: 0.78rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.2px;
    color: #94a3b8; margin-bottom: 8px;
}

.result-fake {
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: #fff; border-radius: 16px; padding: 28px 20px 22px;
    text-align: center; box-shadow: 0 8px 28px rgba(239,68,68,0.35);
}
.result-real {
    background: linear-gradient(135deg, #10b981, #065f46);
    color: #fff; border-radius: 16px; padding: 28px 20px 22px;
    text-align: center; box-shadow: 0 8px 28px rgba(16,185,129,0.35);
}
.result-icon  { font-size: 3.2rem; line-height: 1; margin-bottom: 6px; }
.result-label { font-size: 2rem; font-weight: 800; margin: 0; }
.result-sub   { font-size: 0.9rem; opacity: 0.88; margin-top: 6px; }

.pill-row { display: flex; gap: 10px; margin-top: 18px; justify-content: center; }
.pill { border-radius: 999px; padding: 6px 18px; font-size: 0.85rem; font-weight: 700; display: inline-block; }
.pill-fake { background: rgba(239,68,68,0.15); color: #ef4444; }
.pill-real { background: rgba(16,185,129,0.15); color: #10b981; }

.bar-wrap  { background: #e2e8f0; border-radius: 999px; height: 10px; overflow: hidden; margin: 4px 0 12px; }
.bar-fill  { height: 10px; border-radius: 999px; }

.model-badge {
    display: inline-block; background: #eff6ff; color: #3b82f6;
    border: 1.5px solid #bfdbfe; border-radius: 8px;
    padding: 4px 14px; font-size: 0.82rem; font-weight: 700; margin-bottom: 14px;
}

.chip-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
.chip {
    background: #f8fafc; border: 1.5px solid #e2e8f0;
    border-radius: 10px; padding: 8px 16px;
    flex: 1 1 calc(50% - 10px);
}
.chip .chip-val { font-size: 1.35rem; font-weight: 800; color: #1e293b; }
.chip .chip-lbl { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; }

.acc-ribbon {
    background: #f8fafc; border: 1.5px solid #e2e8f0; border-radius: 12px;
    padding: 14px 20px; display: flex; justify-content: space-between;
    align-items: center; margin-top: 16px; flex-wrap: wrap; gap: 8px;
}
.acc-item { text-align: center; }
.acc-item span { display: block; }
.acc-val { font-size: 1.15rem; font-weight: 800; color: #1e293b; }
.acc-lbl { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; }

div[data-testid="stTextArea"] textarea {
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
    font-size: 0.95rem !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}
div[data-testid="stButton"] button {
    width: 100%; padding: 14px !important;
    border-radius: 12px !important; font-size: 1.05rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
    color: white !important; border: none !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
#  Feature Extraction from Raw Text
# -------------------------------------------------------
def extract_features(text: str) -> dict:
    words     = re.findall(r'\b\w+\b', text)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    word_count    = len(words)
    num_sentences = max(len(sentences), 1)
    unique_words  = len(set(w.lower() for w in words))
    avg_word_len  = (sum(len(w) for w in words) / word_count) if word_count else 0.0

    return {
        "Word_Count":          word_count,
        "Number_of_Sentence":  num_sentences,
        "Unique_Words":        unique_words,
        "Average_Word_Length": round(avg_word_len, 6),
    }


# -------------------------------------------------------
#  Load Data & Train (cached)
# -------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Fake News Detection Dataset.csv")


@st.cache_resource
def train_all_models(_df):
    X = _df.drop(columns=["Label", "ID"], errors="ignore")
    y = _df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    specs = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes":               GaussianNB(),
        "Random Forest":             RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42),
        "Logistic Regression":       LogisticRegression(max_iter=1000, random_state=42),
    }

    trained = {}
    for name, clf in specs.items():
        clf.fit(Xtr, y_train)
        yp = clf.predict(Xte)
        trained[name] = {
            "model":     clf,
            "scaler":    scaler,
            "accuracy":  accuracy_score(y_test, yp),
            "f1":        f1_score(y_test, yp),
            "precision": precision_score(y_test, yp),
            "recall":    recall_score(y_test, yp),
        }
    return trained


df          = load_data()
model_store = train_all_models(df)
MODEL_NAMES = list(model_store.keys())
FEATURES    = ["Word_Count", "Number_of_Sentence", "Unique_Words", "Average_Word_Length"]


# -------------------------------------------------------
#  UI
# -------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>&#128240; Fake News Detector</h1>
    <p>Paste any news article and let machine learning classify it instantly.</p>
</div>
""", unsafe_allow_html=True)

# -- Input card ------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="sec-label">Paste News Article</div>', unsafe_allow_html=True)
news_text = st.text_area(
    label="news_input",
    label_visibility="collapsed",
    placeholder="Type or paste the full news article here...",
    height=200,
)

st.markdown('<div class="sec-label" style="margin-top:18px;">Select Model</div>', unsafe_allow_html=True)
chosen_model = st.selectbox(
    label="model_select",
    label_visibility="collapsed",
    options=MODEL_NAMES,
    index=2,
)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("Predict", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# -- Result ----------------------------------------------
if predict_btn:
    if not news_text.strip():
        st.warning("Please paste a news article before predicting.")
    else:
        feats  = extract_features(news_text)
        res    = model_store[chosen_model]
        clf    = res["model"]
        scaler = res["scaler"]

        X_in = np.array([[feats[f] for f in FEATURES]])
        X_sc = scaler.transform(X_in)
        pred = clf.predict(X_sc)[0]

        if hasattr(clf, "predict_proba"):
            proba     = clf.predict_proba(X_sc)[0]
            conf_fake = float(proba[1])
            conf_real = float(proba[0])
        else:
            conf_fake = 1.0 if pred == 1 else 0.0
            conf_real = 1.0 - conf_fake

        # -- Result banner --------------------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="model-badge">Model: {chosen_model}</div>', unsafe_allow_html=True)

        if pred == 1:
            st.markdown(f"""
            <div class="result-fake">
                <div class="result-icon">&#128680;</div>
                <p class="result-label">FAKE NEWS</p>
                <p class="result-sub">This article is likely fabricated or misleading.</p>
                <div class="pill-row">
                    <span class="pill" style="background:rgba(255,255,255,0.2);color:#fff;">Fake {conf_fake*100:.1f}%</span>
                    <span class="pill" style="background:rgba(255,255,255,0.1);color:#ffffffbb;">Real {conf_real*100:.1f}%</span>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-real">
                <div class="result-icon">&#9989;</div>
                <p class="result-label">REAL NEWS</p>
                <p class="result-sub">This article appears to be legitimate.</p>
                <div class="pill-row">
                    <span class="pill" style="background:rgba(255,255,255,0.2);color:#fff;">Real {conf_real*100:.1f}%</span>
                    <span class="pill" style="background:rgba(255,255,255,0.1);color:#ffffffbb;">Fake {conf_fake*100:.1f}%</span>
                </div>
            </div>""", unsafe_allow_html=True)

        # -- Confidence bars ------------------------------
        st.markdown("<br><b>Confidence</b>", unsafe_allow_html=True)

        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.markdown(f'<div class="bar-wrap"><div class="bar-fill" style="width:{conf_fake*100:.1f}%;background:#ef4444;"></div></div>', unsafe_allow_html=True)
        with col_r:
            st.markdown(f'<span style="font-weight:700;color:#ef4444;">Fake {conf_fake*100:.1f}%</span>', unsafe_allow_html=True)

        col_l2, col_r2 = st.columns([3, 1])
        with col_l2:
            st.markdown(f'<div class="bar-wrap"><div class="bar-fill" style="width:{conf_real*100:.1f}%;background:#10b981;"></div></div>', unsafe_allow_html=True)
        with col_r2:
            st.markdown(f'<span style="font-weight:700;color:#10b981;">Real {conf_real*100:.1f}%</span>', unsafe_allow_html=True)

        # -- Extracted features ---------------------------
        st.markdown("<br><b>Extracted Article Features</b>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="chip-grid">
            <div class="chip"><div class="chip-val">{feats["Word_Count"]}</div><div class="chip-lbl">Word Count</div></div>
            <div class="chip"><div class="chip-val">{feats["Number_of_Sentence"]}</div><div class="chip-lbl">Sentences</div></div>
            <div class="chip"><div class="chip-val">{feats["Unique_Words"]}</div><div class="chip-lbl">Unique Words</div></div>
            <div class="chip"><div class="chip-val">{feats["Average_Word_Length"]:.2f}</div><div class="chip-lbl">Avg Word Length</div></div>
        </div>""", unsafe_allow_html=True)

        # -- Model metrics ribbon -------------------------
        st.markdown(f"""
        <div class="acc-ribbon">
            <div class="acc-item"><span class="acc-val">{res["accuracy"]*100:.2f}%</span><span class="acc-lbl">Accuracy</span></div>
            <div class="acc-item"><span class="acc-val">{res["f1"]:.4f}</span><span class="acc-lbl">F1 Score</span></div>
            <div class="acc-item"><span class="acc-val">{res["precision"]:.4f}</span><span class="acc-lbl">Precision</span></div>
            <div class="acc-item"><span class="acc-val">{res["recall"]:.4f}</span><span class="acc-lbl">Recall</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# -- All-models accuracy comparison ----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="sec-label">All Models - Accuracy Comparison</div>', unsafe_allow_html=True)
for name, data in model_store.items():
    acc_pct    = data["accuracy"] * 100
    is_chosen  = name == chosen_model
    bar_color  = "#3b82f6" if is_chosen else "#cbd5e1"
    lbl_style  = "font-weight:800;color:#1e293b;" if is_chosen else "color:#64748b;"
    prefix     = "* " if is_chosen else ""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">'
        f'<span style="width:230px;font-size:0.85rem;{lbl_style}">{prefix}{name}</span>'
        f'<div class="bar-wrap" style="flex:1;margin:0;">'
        f'<div class="bar-fill" style="width:{acc_pct:.1f}%;background:{bar_color};"></div></div>'
        f'<span style="min-width:52px;text-align:right;font-size:0.85rem;font-weight:700;color:#1e293b;">{acc_pct:.2f}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
st.markdown('</div>', unsafe_allow_html=True)

# -- Footer ----------------------------------------------
st.markdown(
    '<p style="text-align:center;color:#94a3b8;font-size:0.78rem;padding-top:4px;">'
    'KNN &nbsp;|&nbsp; Naive Bayes &nbsp;|&nbsp; Random Forest &nbsp;|&nbsp; Logistic Regression'
    '</p>',
    unsafe_allow_html=True,
)
