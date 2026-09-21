import html
import numpy as np
import joblib
import streamlit as st
from pathlib import Path
from collections import Counter
from pythainlp.tokenize import word_tokenize

# ต้องมีฟังก์ชันชื่อนี้อยู่ตรงนี้เป๊ะๆ เพราะ vectorizer.joblib
# ถูก pickle ไว้พร้อมการอ้างอิงถึงฟังก์ชันนี้ตอนเทรน (ห้ามลบ/ห้ามเปลี่ยนชื่อ)
def thai_tokenizer(text):
    return word_tokenize(text, engine="newmm")

# ----------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Thai Threat Message Classifier",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# Custom styling — trustworthy core, playful accents, blue-teal palette
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #0f2438;
        --muted: #5b7285;
        --teal: #0ea5a3;
        --teal-dark: #0b8482;
        --blue: #2563eb;
        --blue-light: #eaf2ff;
        --surface: #ffffff;
        --page-bg: #f4f9f9;
        --border: #dbe7e6;
        --danger: #e0435a;
        --danger-bg: #fdeef0;
        --warn: #e08a1e;
        --warn-bg: #fdf4e6;
        --safe: #16a35c;
        --safe-bg: #eafbf1;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Noto Sans Thai', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background: var(--page-bg);
    }

    /* ---------- Header ---------- */
    .main-header {
        text-align: center;
        padding: 1.6rem 0 0.8rem 0;
    }
    .main-header .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--teal-dark);
        background: #e4f7f6;
        border: 1px solid #bfe9e7;
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        margin-bottom: 0.7rem;
    }
    .main-header h1 {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        color: var(--ink);
        letter-spacing: -0.02em;
    }
    .main-header h1 .accent {
        color: var(--teal);
    }
    .main-header p {
        color: var(--muted);
        font-size: 0.95rem;
        font-weight: 500;
        max-width: 32rem;
        margin: 0 auto;
    }

    /* ---------- Input box ---------- */
    .stTextArea textarea {
        border-radius: 14px !important;
        border: 1.5px solid var(--border) !important;
        font-size: 1rem;
        padding: 0.9rem !important;
        background: var(--surface) !important;
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
        caret-color: var(--ink) !important;
        box-shadow: 0 1px 2px rgba(15,36,56,0.04);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .stTextArea textarea::placeholder {
        color: var(--muted) !important;
        -webkit-text-fill-color: var(--muted) !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 3px rgba(14,165,163,0.15) !important;
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
    }
    .stTextArea label p {
        font-weight: 700 !important;
        color: var(--ink) !important;
    }

    /* ---------- Buttons ---------- */
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 0.65rem 0;
        font-weight: 700;
        font-size: 0.98rem;
        background: linear-gradient(180deg, var(--teal) 0%, var(--teal-dark) 100%);
        color: white;
        border: none;
        box-shadow: 0 2px 6px rgba(14,165,163,0.28);
        transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.04);
        box-shadow: 0 4px 10px rgba(14,165,163,0.34);
    }
    div.stButton > button:active {
        transform: translateY(0);
    }

    div[data-testid="column"]:nth-of-type(2) div.stButton > button {
        background: var(--surface);
        color: var(--muted);
        border: 1.5px solid var(--border);
        box-shadow: none;
    }
    div[data-testid="column"]:nth-of-type(2) div.stButton > button:hover {
        border-color: var(--teal);
        color: var(--teal-dark);
    }

    /* ---------- Result card ---------- */
    @keyframes riseIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-card {
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin-top: 1.4rem;
        border: 1.5px solid var(--border);
        background: var(--surface);
        box-shadow: 0 4px 16px rgba(15,36,56,0.06);
        animation: riseIn 0.28s ease;
    }
    .result-card.danger { border-color: #f4c2cb; background: var(--danger-bg); }
    .result-card.warn   { border-color: #f5dba8; background: var(--warn-bg); }
    .result-card.safe   { border-color: #b7ecce; background: var(--safe-bg); }

    .result-eyebrow {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.3rem;
    }
    .result-label {
        font-size: 1.3rem;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .confidence-row {
        display: flex;
        justify-content: space-between;
        margin-top: 0.55rem;
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--ink);
    }
    .bar-bg {
        background: #e9eff0;
        border-radius: 999px;
        height: 9px;
        margin-top: 4px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--teal), var(--blue));
        transition: width 0.4s ease;
    }

    /* ---------- Highlighted words (interpretability) ---------- */
    .highlight-box {
        margin-top: 1rem;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        background: var(--blue-light);
        border: 1px solid #cfe0fb;
        font-size: 0.95rem;
        line-height: 1.9;
        color: var(--ink);
    }
    .highlight-box .title {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--blue);
        margin-bottom: 0.5rem;
        display: block;
    }
    mark.word-hit {
        background: linear-gradient(180deg, transparent 60%, #ffd76a 60%);
        padding: 0 1px;
        border-radius: 2px;
        font-weight: 700;
        color: var(--ink);
    }

    /* ---------- History ---------- */
    .history-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.6rem;
        padding: 0.55rem 0.8rem;
        margin-bottom: 0.4rem;
        border-radius: 10px;
        background: var(--surface);
        border: 1px solid var(--border);
        font-size: 0.85rem;
    }
    .history-item .txt {
        color: var(--muted);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        flex: 1;
    }
    .history-tag {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        white-space: nowrap;
    }
    .history-tag.danger { background: var(--danger-bg); color: var(--danger); }
    .history-tag.warn   { background: var(--warn-bg); color: var(--warn); }
    .history-tag.safe   { background: var(--safe-bg); color: var(--safe); }

    .stat-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--ink);
        margin-top: 0.7rem;
    }
    .stat-bar-bg {
        background: #e9eff0;
        border-radius: 999px;
        height: 10px;
        margin-top: 4px;
        overflow: hidden;
    }
    .stat-bar-fill {
        height: 100%;
        border-radius: 999px;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Load model artifacts (cached so they only load once per session)
# ----------------------------------------------------------------------
MODEL_DIR = Path(__file__).parent  # put the .joblib files next to this script

@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load(MODEL_DIR / "vectorizer.joblib")
    selector = joblib.load(MODEL_DIR / "selector.joblib")
    model = joblib.load(MODEL_DIR / "model.joblib")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    return vectorizer, selector, model, label_encoder

try:
    vectorizer, selector, model, label_encoder = load_artifacts()
    load_error = None
except Exception as e:
    load_error = str(e)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>Thai Threat <span class="accent">Classifier</span></h1>
        <p>วิเคราะห์ข้อความภาษาไทย เพื่อตรวจจับการดูหมิ่น / คุกคามทางเพศ / ความแตกต่างทางสังคม</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.error(
        f"โหลดไฟล์โมเดลไม่สำเร็จ: {load_error}\n\n"
        "ตรวจสอบว่าไฟล์ vectorizer.joblib, selector.joblib, model.joblib, "
        "label_encoder.joblib อยู่โฟลเดอร์เดียวกับ app.py"
    )
    st.stop()

# ----------------------------------------------------------------------
# Label metadata
# ----------------------------------------------------------------------
LABEL_META = {
    "insult": {"th": "ดูหมิ่น", "icon": "⚠️", "tone": "danger"},
    "sexual harassment": {"th": "คุกคามทางเพศ", "icon": "🚫", "tone": "danger"},
    "social differentiation": {"th": "ความแตกต่างทางสังคม", "icon": "⚡", "tone": "warn"},
    "normal": {"th": "ข้อความปกติ", "icon": "✅", "tone": "safe"},
}

TONE_COLOR = {"danger": "#e0435a", "warn": "#e08a1e", "safe": "#16a35c"}

def label_info(raw_label: str):
    meta = LABEL_META.get(raw_label, {"th": raw_label, "icon": "🏷️", "tone": "warn"})
    return meta["th"], meta["icon"], meta["tone"]

# ----------------------------------------------------------------------
# Session state: history of checks made in this session
# ----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # each item: dict(text, label_th, icon, tone, confidence)

# ----------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------
def clear_text():
    st.session_state["user_text"] = ""

text_input = st.text_area(
    "พิมพ์ข้อความที่ต้องการตรวจสอบ",
    placeholder="เช่น ข้อความจากโพสต์หรือคอมเมนต์บน X/Twitter ...",
    height=140,
    key="user_text",
)

col1, col2 = st.columns([3, 1])
with col1:
    analyze = st.button("🔍 วิเคราะห์ข้อความ", use_container_width=True)
with col2:
    clear = st.button("ล้างข้อความ", use_container_width=True, on_click=clear_text)

# ----------------------------------------------------------------------
# Helper: find words that pushed the prediction the most (interpretability)
# ----------------------------------------------------------------------
def get_influential_words(text: str, X_vec, top_n: int = 8):
    """
    ประมาณคำที่มีผลต่อการตัดสินใจของโมเดล โดยดูจาก
    (ค่า TF-IDF ของคำในข้อความนี้) x (ความสำคัญของฟีเจอร์นั้นในโมเดล)
    ยิ่งค่าสูง ยิ่งมีอิทธิพลต่อผลลัพธ์ที่โมเดลทำนายออกมา
    """
    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
        support_mask = selector.get_support()
        selected_names = feature_names[support_mask]

        tfidf_full = X_vec.toarray()[0]
        tfidf_selected = tfidf_full[support_mask]

        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_)
        else:
            importances = np.ones(len(selected_names))

        contribution = tfidf_selected * importances
        nonzero = np.where(contribution > 0)[0]
        if len(nonzero) == 0:
            return []

        top_idx = nonzero[np.argsort(contribution[nonzero])[::-1][:top_n]]
        return [selected_names[i] for i in top_idx]
    except Exception:
        return []


def render_highlighted_text(text: str, hit_words) -> str:
    """ตัดคำข้อความเดิมด้วย tokenizer ตัวเดียวกับตอนเทรน แล้วห่อคำที่มีอิทธิพลด้วย <mark>"""
    hit_set = set(w.strip() for w in hit_words if w.strip())
    if not hit_set:
        return html.escape(text)

    tokens = thai_tokenizer(text)
    parts = []
    for tok in tokens:
        safe_tok = html.escape(tok)
        if tok.strip() in hit_set:
            parts.append(f'<mark class="word-hit">{safe_tok}</mark>')
        else:
            parts.append(safe_tok)
    return "".join(parts)

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if analyze:
    if not text_input.strip():
        st.warning("กรุณาพิมพ์ข้อความก่อนวิเคราะห์")
    else:
        with st.spinner("กำลังวิเคราะห์..."):
            X_vec = vectorizer.transform([text_input])
            X_sel = selector.transform(X_vec)
            pred_idx = model.predict(X_sel)[0]
            pred_label = label_encoder.inverse_transform([pred_idx])[0]

            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_sel)[0]

            influential_words = get_influential_words(text_input, X_vec)

        label_th, icon, tone = label_info(pred_label)
        top_conf = float(np.max(proba) * 100) if proba is not None else None

        # บันทึกลงประวัติของเซสชันนี้ (ไม่ได้เก็บถาวรข้ามเซสชัน)
        st.session_state.history.insert(
            0,
            {
                "text": text_input[:60],
                "label_th": label_th,
                "icon": icon,
                "tone": tone,
                "confidence": top_conf,
            },
        )
        st.session_state.history = st.session_state.history[:15]

        st.markdown(
            f"""
            <div class="result-card {tone}">
                <div class="result-eyebrow">ผลการวิเคราะห์</div>
                <div class="result-label">{icon} {label_th}</div>
            """,
            unsafe_allow_html=True,
        )

        if proba is not None:
            st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)
            order = np.argsort(proba)[::-1]
            for i in order:
                cls_raw = label_encoder.inverse_transform([i])[0]
                cls_th, _, _ = label_info(cls_raw)
                pct = proba[i] * 100
                st.markdown(
                    f"""
                    <div class="confidence-row">
                        <span>{cls_th}</span><span>{pct:.1f}%</span>
                    </div>
                    <div class="bar-bg"><div class="bar-fill" style="width:{pct}%;"></div></div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----- ไฮไลต์คำที่มีผลต่อการตัดสินใจของโมเดล -----
        #highlighted_html = render_highlighted_text(text_input, influential_words)
        # st.markdown(
        #     f"""
        #     <div class="highlight-box">
        #         <span class="title">คำที่มีผลต่อผลการวิเคราะห์</span>
        #         {highlighted_html}
        #     </div>
        #     """,
        #     unsafe_allow_html=True,
        # )
        # if not influential_words:
        #     st.caption("ไม่พบคำที่มีน้ำหนักชัดเจนพอจะไฮไลต์สำหรับข้อความนี้")

        # st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# History + mini stats chart (เฉพาะในเซสชันนี้)
# ----------------------------------------------------------------------
if st.session_state.history:
    st.markdown("<div style='margin-top:1.6rem;'></div>", unsafe_allow_html=True)

    with st.expander(f"สรุปสถิติจากประวัติที่ตรวจสอบ ({len(st.session_state.history)} ข้อความ)", expanded=False):
        label_counts = Counter(item["label_th"] for item in st.session_state.history)
        total = sum(label_counts.values())

        # เตรียมสีของแต่ละ label จาก tone ล่าสุดที่เจอ
        label_tone_lookup = {}
        for item in st.session_state.history:
            label_tone_lookup.setdefault(item["label_th"], item["tone"])

        st.markdown("**สัดส่วนประเภทที่พบบ่อย**")
        for label_th, count in label_counts.most_common():
            pct = count / total * 100
            tone = label_tone_lookup.get(label_th, "warn")
            color = TONE_COLOR.get(tone, "#0ea5a3")
            st.markdown(
                f"""
                <div class="stat-row"><span>{label_th}</span><span>{count} ครั้ง ({pct:.0f}%)</span></div>
                <div class="stat-bar-bg"><div class="stat-bar-fill" style="width:{pct}%; background:{color};"></div></div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        st.markdown("**รายการล่าสุด**")
        for item in st.session_state.history:
            conf_str = f"{item['confidence']:.0f}%" if item["confidence"] is not None else "-"
            st.markdown(
                f"""
                <div class="history-item">
                    <span class="txt">{html.escape(item['text'])}</span>
                    <span class="history-tag {item['tone']}">{item['icon']} {item['label_th']} · {conf_str}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("ล้างประวัติ", use_container_width=True):
            st.session_state.history = []
            st.rerun()

st.markdown(
    "<p style='text-align:center; color:#9ca3af; font-size:0.8rem; margin-top:2rem;'>"
    "</p>",
    unsafe_allow_html=True,
)
