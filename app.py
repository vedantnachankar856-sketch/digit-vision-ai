import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import pickle
import time
import io
import base64

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Vision AI",
    page_icon="🔢",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

/* Global dark theme */
html, body, [class*="css"] {
    background-color: #0a0a0f !important;
    color: #e0e0e0 !important;
}
.stApp {
    background: radial-gradient(ellipse at 20% 50%, #0d1b2a 0%, #0a0a0f 60%, #0d0a1a 100%);
    min-height: 100vh;
}

/* Hide default streamlit elements */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1rem !important; max-width: 800px;}

/* Animated title */
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #00d4ff, #7c3aed, #00d4ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s linear infinite;
    letter-spacing: 3px;
    margin-bottom: 0;
}
@keyframes shimmer {
    0% {background-position: 0% center;}
    100% {background-position: 200% center;}
}

.hero-sub {
    font-family: 'Share Tech Mono', monospace;
    text-align: center;
    color: #00d4ff88;
    font-size: 0.85rem;
    letter-spacing: 4px;
    margin-top: 4px;
    text-transform: uppercase;
}

/* Glowing divider */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #00d4ff, #7c3aed, #00d4ff, transparent);
    margin: 1.5rem 0;
    animation: pulse-glow 2s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%, 100% {opacity: 0.5;}
    50% {opacity: 1;}
}

/* Upload zone */
.upload-zone {
    border: 1px solid #00d4ff33;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    background: linear-gradient(135deg, #0d1b2a88, #0d0a1a88);
    backdrop-filter: blur(10px);
    margin: 1rem 0;
    transition: all 0.3s ease;
}

/* Prediction card */
.pred-card {
    background: linear-gradient(135deg, #0d1b2a, #130a2a);
    border: 1px solid #00d4ff44;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: card-appear 0.6s ease-out;
    box-shadow: 0 0 40px #00d4ff22, inset 0 0 40px #7c3aed11;
}
@keyframes card-appear {
    from {opacity: 0; transform: translateY(30px) scale(0.95);}
    to {opacity: 1; transform: translateY(0) scale(1);}
}

/* Corner decorations */
.pred-card::before, .pred-card::after {
    content: '';
    position: absolute;
    width: 60px; height: 60px;
    border-color: #00d4ff;
    border-style: solid;
}
.pred-card::before {top: 10px; left: 10px; border-width: 2px 0 0 2px; border-radius: 4px 0 0 0;}
.pred-card::after {bottom: 10px; right: 10px; border-width: 0 2px 2px 0; border-radius: 0 0 4px 0;}

.pred-digit {
    font-family: 'Orbitron', monospace;
    font-size: 6rem;
    font-weight: 900;
    background: linear-gradient(180deg, #ffffff, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    animation: digit-pop 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
@keyframes digit-pop {
    0% {transform: scale(0); opacity: 0;}
    70% {transform: scale(1.15);}
    100% {transform: scale(1); opacity: 1;}
}

.pred-label {
    font-family: 'Share Tech Mono', monospace;
    color: #00d4ffaa;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.conf-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #7c3aed;
}

/* Confidence bar */
.conf-bar-wrap {
    background: #ffffff11;
    border-radius: 50px;
    height: 8px;
    margin: 0.5rem 0 1.5rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #7c3aed, #00d4ff);
    animation: bar-grow 1s ease-out forwards;
    transform-origin: left;
}
@keyframes bar-grow {
    from {width: 0 !important;}
}

/* Top predictions */
.top-pred-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0;
    animation: slide-in 0.4s ease-out backwards;
}
@keyframes slide-in {
    from {opacity: 0; transform: translateX(-20px);}
    to {opacity: 1; transform: translateX(0);}
}
.top-pred-digit-badge {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    color: #00d4ff;
    background: #00d4ff15;
    border: 1px solid #00d4ff33;
    border-radius: 6px;
    padding: 2px 10px;
    min-width: 36px;
    text-align: center;
}
.top-pred-bar-outer {
    flex: 1;
    background: #ffffff0d;
    border-radius: 50px;
    height: 6px;
    overflow: hidden;
}
.top-pred-bar-inner {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #7c3aed88, #00d4ff88);
}
.top-pred-pct {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #ffffff66;
    min-width: 45px;
    text-align: right;
}

/* Info badge */
.info-badge {
    display: inline-block;
    background: #00d4ff11;
    border: 1px solid #00d4ff33;
    border-radius: 20px;
    padding: 4px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #00d4ff88;
    margin: 4px;
}

/* Scanning animation */
.scan-line {
    position: relative;
    overflow: hidden;
    border-radius: 12px;
}
.scan-line::after {
    content: '';
    position: absolute;
    top: -100%;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
    animation: scan 1.5s ease-in-out;
}
@keyframes scan {
    0% {top: -5%;}
    100% {top: 105%;}
}

/* Section headers */
.section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #00d4ff66;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #ffffff11;
}
.footer a {
    color: #00d4ff88;
    text-decoration: none;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    margin: 0 12px;
    transition: color 0.2s;
}
.footer a:hover {color: #00d4ff;}

/* File uploader styling */
[data-testid="stFileUploader"] {
    background: #0d1b2a44 !important;
    border: 1px dashed #00d4ff44 !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] label {
    color: #00d4ffaa !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* Image display */
[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid #00d4ff33;
    box-shadow: 0 0 20px #00d4ff22;
}

/* Streamlit buttons */
.stButton > button {
    background: linear-gradient(135deg, #0d1b2a, #130a2a) !important;
    border: 1px solid #00d4ff66 !important;
    color: #00d4ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 2px !important;
    border-radius: 8px !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    border-color: #00d4ff !important;
    box-shadow: 0 0 15px #00d4ff44 !important;
    transform: translateY(-1px) !important;
}

/* Spinner */
[data-testid="stSpinner"] {color: #00d4ff !important;}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("prediction.pkl", "rb") as f:
        return pickle.load(f)

# Class names for 10-class output (MNIST style)
CLASS_NAMES = [str(i) for i in range(10)]

# ── Helper: preprocess image ──────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    # Convert to grayscale
    img = img.convert("L")
    # Resize to 28x28
    img = img.resize((28, 28), Image.LANCZOS)
    # Convert to numpy
    arr = np.array(img, dtype=np.float32)
    # Normalize
    arr = arr / 255.0
    # Reshape to (1, 28, 28, 1)
    arr = arr.reshape(1, 28, 28, 1)
    return arr


# ── UI ────────────────────────────────────────────────────────────────────────

# Hero header
st.markdown('<div class="hero-title">DIGIT VISION AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">CNN · Keras · 10-Class Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

# Info badges
st.markdown("""
<div style="text-align:center; margin-bottom:1.5rem;">
    <span class="info-badge">🧠 CNN Architecture</span>
    <span class="info-badge">📐 28×28 Input</span>
    <span class="info-badge">🎯 10 Classes (0–9)</span>
    <span class="info-badge">⚡ Keras 3.13</span>
</div>
""", unsafe_allow_html=True)

# Load model
try:
    model = load_model()
    st.markdown("""
    <div style="text-align:center; margin-bottom:1rem;">
        <span style="font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:#00ff8844;">
            ✅ MODEL LOADED SUCCESSFULLY
        </span>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"⚠️ Could not load model: {e}")
    st.stop()

# Upload section
st.markdown('<div class="section-header">// upload image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop a digit image here (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    # Show image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        img = Image.open(uploaded_file)
        st.markdown('<div class="scan-line">', unsafe_allow_html=True)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("⚡ ANALYZE IMAGE", use_container_width=True)

    if predict_btn:
        with st.spinner("🔍 Running neural network..."):
            time.sleep(0.6)  # dramatic pause
            try:
                arr = preprocess_image(img)
                preds = model.predict(arr, verbose=0)[0]
                pred_class = int(np.argmax(preds))
                confidence = float(preds[pred_class]) * 100
                pred_label = CLASS_NAMES[pred_class]

                # Top 5 predictions
                top5_idx = np.argsort(preds)[::-1][:5]

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">// prediction result</div>', unsafe_allow_html=True)

        # Main result card
        conf_color = "#00ff88" if confidence >= 80 else "#ffaa00" if confidence >= 50 else "#ff4444"
        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-label">PREDICTED CLASS</div>
            <div class="pred-digit">{pred_label}</div>
            <div style="margin:0.8rem 0;">
                <span class="pred-label">CONFIDENCE &nbsp;</span>
                <span class="conf-value" style="color:{conf_color};">{confidence:.1f}%</span>
            </div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{confidence}%;"></div>
            </div>
            <div class="pred-label" style="font-size:0.65rem; margin-top:0.5rem;">
                {"🎯 HIGH CONFIDENCE" if confidence >= 80 else "⚠️ MODERATE CONFIDENCE" if confidence >= 50 else "❓ LOW CONFIDENCE"}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Top 5 breakdown
        st.markdown('<br/>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">// top 5 predictions</div>', unsafe_allow_html=True)

        for i, idx in enumerate(top5_idx):
            pct = float(preds[idx]) * 100
            delay = i * 0.1
            bar_color = "linear-gradient(90deg, #7c3aed, #00d4ff)" if i == 0 else "linear-gradient(90deg, #7c3aed44, #00d4ff44)"
            st.markdown(f"""
            <div class="top-pred-row" style="animation-delay:{delay}s;">
                <div class="top-pred-digit-badge">{CLASS_NAMES[idx]}</div>
                <div class="top-pred-bar-outer">
                    <div class="top-pred-bar-inner" style="width:{pct}%; background:{bar_color};"></div>
                </div>
                <div class="top-pred-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Preprocessed image insight
        st.markdown('<br/>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">// preprocessed input (28×28)</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            processed = preprocess_image(img)
            processed_display = (processed[0, :, :, 0] * 255).astype(np.uint8)
            pil_processed = Image.fromarray(processed_display, mode='L').resize((140, 140), Image.NEAREST)
            st.image(pil_processed, caption="Model input (28×28 → displayed 140×140)", use_column_width=False)

else:
    # Placeholder when nothing uploaded
    st.markdown("""
    <div class="upload-zone">
        <div style="font-size:3rem; margin-bottom:1rem;">🔢</div>
        <div style="font-family:'Share Tech Mono',monospace; color:#00d4ff88; font-size:0.85rem; letter-spacing:2px;">
            UPLOAD A HANDWRITTEN DIGIT IMAGE
        </div>
        <div style="font-family:'Share Tech Mono',monospace; color:#ffffff33; font-size:0.7rem; margin-top:0.5rem; letter-spacing:1px;">
            PNG · JPG · JPEG &nbsp;|&nbsp; Best results: white digit on black background
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tips
    st.markdown('<br/>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">// how it works</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    tips = [
        ("📤", "UPLOAD", "Upload any image of a handwritten digit (0–9)"),
        ("⚡", "ANALYZE", "CNN processes image through Conv2D layers"),
        ("🎯", "PREDICT", "Get confidence scores for all 10 digit classes"),
    ]
    for col, (icon, title, desc) in zip(cols, tips):
        with col:
            st.markdown(f"""
            <div style="background:#0d1b2a44; border:1px solid #00d4ff22; border-radius:12px; padding:1rem; text-align:center; height:140px;">
                <div style="font-size:1.8rem;">{icon}</div>
                <div style="font-family:'Orbitron',monospace; font-size:0.65rem; color:#00d4ffaa; letter-spacing:2px; margin:6px 0;">{title}</div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#ffffff55; line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#ffffff22; letter-spacing:2px; margin-bottom:0.8rem;">
        BUILT BY VEDANT NACHANKAR
    </div>
    <a href="https://github.com/vedantnachankar856-sketch" target="_blank">⚡ GitHub</a>
    <a href="https://www.linkedin.com/in/vedant-nachankar-6396783b1" target="_blank">💼 LinkedIn</a>
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#ffffff11; margin-top:0.8rem; letter-spacing:1px;">
        KERAS CNN · STREAMLIT · PYTHON
    </div>
</div>
""", unsafe_allow_html=True)
