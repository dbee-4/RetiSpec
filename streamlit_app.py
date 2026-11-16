import io
import os
import gc
import numpy as np
import torch
from PIL import Image
import streamlit as st

import config as cfg
from model import build_model
from preprocessing import preprocess_for_model, preprocess_for_tta
from tta import TTAPredictor
from xai_gradcam import create_gradcam_engine, compute_gradcam
from xai_gradientxinput import compute_gradient_x_input

# ── Streamlit Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="RetiSpec - Explainable AI DR Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for RetiSpec Dark Glassmorphism ───────────────────────────────
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #111827 100%);
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Header Styling */
    .header-box {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        margin-bottom: 1.5rem;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .header-sub {
        font-size: 1.05rem;
        color: #94a3b8;
    }

    /* Diagnosis Banner Card */
    .report-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    .report-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }
    .badge-no-dr { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .badge-mild { background-color: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid #eab308; }
    .badge-moderate { background-color: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid #f97316; }
    .badge-severe { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .badge-proliferative { background-color: rgba(225, 29, 72, 0.25); color: #fb7185; border: 1px solid #e11d48; }

    /* XAI Visual Cards */
    .xai-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .xai-title {
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.8px;
        color: #94a3b8;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .xai-caption {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.4rem;
        line-height: 1.3;
    }
</style>
""", unsafe_allow_html=True)

# ── Metadata Constants ────────────────────────────────────────────────────────
LABEL_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
SEVERITY_DESC = [
    "No visible diabetic retinopathy.",
    "Small areas of swelling in retinal blood vessels (microaneurysms).",
    "Leakage in blood vessels impacting the retina.",
    "Many blocked blood vessels, depriving the retina of blood flow.",
    "Advanced stage with growth of new, fragile blood vessels."
]
BADGE_CLASSES = [
    "badge-no-dr",
    "badge-mild",
    "badge-moderate",
    "badge-severe",
    "badge-proliferative"
]
BAR_COLORS = ["#10b981", "#eab308", "#f97316", "#ef4444", "#e11d48"]


# ── Model & Engine Caching ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RetiSpec Deep Learning Engine...")
def load_retispec_engines():
    """Initializes and caches ConvNeXt-Tiny and XAI generators."""
    torch.set_num_threads(2)
    device = torch.device("cpu")

    # 1. Build ConvNeXt-Tiny
    model = build_model(pretrained=False, device=device)
    if os.path.exists(cfg.BEST_MODEL_PATH):
        ckpt = torch.load(cfg.BEST_MODEL_PATH, map_location=device)
        state_dict = ckpt.get("ema_state", ckpt.get("model_state", ckpt.get("model_state_dict", ckpt)))
        model.load_state_dict(state_dict)
        del ckpt, state_dict
        gc.collect()
    model.eval()

    # 2. XAI & TTA Engines
    cam_engine = create_gradcam_engine(model)
    tta_engine = TTAPredictor(model, device=device)

    return model, cam_engine, tta_engine, device


model, cam_engine, tta_engine, device = load_retispec_engines()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <div class="header-title">👁️ RetiSpec</div>
    <div class="header-sub">Explainable AI based Diabetic Retinopathy Detection</div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar: Upload & Quick Samples ───────────────────────────────────────────
with st.sidebar:
    st.header("1. Input Retinal Fundus")
    
    # Pre-loaded sample images
    sample_dir = os.path.join(cfg.BASE_DIR, "test images")
    sample_options = ["None (Upload my own)"]
    sample_files = []
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))]
        sample_options += sample_files

    selected_sample = st.selectbox("Or choose a sample fundus image:", sample_options)
    
    uploaded_file = st.file_uploader(
        "Upload retinal fundus photograph",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload standard digital retinal fundus photograph."
    )

    st.markdown("---")
    st.markdown("**Pipeline Configuration**")
    st.info("⚙️ **Inference**: ConvNeXt-Tiny\n\n🛡️ **TTA**: 6-Fold Ensemble (Always Active)\n\n🔬 **XAI**: Grad-CAM++ & Gradient × Input")


# ── Resolve Input Image ───────────────────────────────────────────────────────
pil_img = None

if uploaded_file is not None:
    try:
        pil_img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error loading uploaded image: {e}")
elif selected_sample != "None (Upload my own)" and sample_files:
    sample_path = os.path.join(sample_dir, selected_sample)
    try:
        pil_img = Image.open(sample_path).convert("RGB")
    except Exception as e:
        st.error(f"Error loading sample image: {e}")


# ── Main Content Area ─────────────────────────────────────────────────────────
if pil_img is None:
    st.info("👋 Welcome! Please upload a retinal fundus image from the left sidebar or select one of the built-in test samples to begin analysis.")
else:
    # Top Section: Two columns (Original Fundus vs Prediction Summary)
    col_input, col_report = st.columns([1, 1.4], gap="large")

    with col_input:
        st.markdown("<div class='xai-title'>Original Fundus Image</div>", unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)

    with col_report:
        with st.spinner("Analyzing fundus with ConvNeXt-Tiny & Dual XAI..."):
            # 1. Preprocessing
            input_tensor, vis_img = preprocess_for_model(pil_img, device=device)

            # 2. Test-Time Augmentation (6 variants)
            img_preprocessed = preprocess_for_tta(pil_img)
            avg_probs, pred_idx, _ = tta_engine.predict_with_tta(img_preprocessed, num_augments=6)
            conf = float(avg_probs[pred_idx])

            # 3. Dual XAI Visualizations
            # Form 1: Grad-CAM++
            cam_overlay = compute_gradcam(cam_engine, input_tensor, pred_idx, vis_img)
            # Form 2: Gradient x Input
            grad_x_input_overlay = compute_gradient_x_input(model, input_tensor, pred_idx, vis_img)

        # Diagnostic Banner
        pred_label = LABEL_NAMES[pred_idx]
        badge_class = BADGE_CLASSES[pred_idx]
        description = SEVERITY_DESC[pred_idx]

        if pred_idx == 0:
            explanation = "This person is not affected by diabetic retinopathy."
        else:
            explanation = f"This person is affected by {pred_label.lower()} diabetic retinopathy."

        st.markdown(f"""
        <div class="report-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <span class="report-badge {badge_class}">{pred_label}</span>
                <span style="font-size: 1.25rem; font-weight: 700; color: #38bdf8;">{conf * 100:.2f}% Confidence</span>
            </div>
            <div style="font-size: 0.95rem; color: #cbd5e1; margin-bottom: 0.6rem;">{description}</div>
            <div style="background: rgba(15, 23, 42, 0.6); border-left: 3px solid #38bdf8; padding: 0.6rem 0.8rem; border-radius: 4px; font-size: 0.9rem; color: #e2e8f0; font-style: italic;">
                {explanation}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Class Probability Distribution
        st.markdown("<div class='xai-title'>Severity Class Distribution</div>", unsafe_allow_html=True)
        for i, (name, prob) in enumerate(zip(LABEL_NAMES, avg_probs)):
            prob_pct = float(prob) * 100
            bar_col = BAR_COLORS[i]
            st.markdown(f"""
            <div style="margin-bottom: 0.4rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #cbd5e1; margin-bottom: 2px;">
                    <span>{name}</span>
                    <span style="font-weight: 600;">{prob_pct:.2f}%</span>
                </div>
                <div style="background: rgba(51, 65, 85, 0.4); border-radius: 9999px; height: 7px; overflow: hidden;">
                    <div style="background: {bar_col}; width: {prob_pct}%; height: 100%; border-radius: 9999px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Bottom Section: 3 Side-by-Side XAI Cards
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 2. Explainable AI Diagnostic Visualizations")
    
    col_pre, col_cam, col_gxi = st.columns(3, gap="medium")

    with col_pre:
        st.markdown("""
        <div class="xai-card">
            <div class="xai-title">Preprocessed Input</div>
        """, unsafe_allow_html=True)
        st.image(vis_img, use_container_width=True)
        st.markdown("""
            <div class="xai-caption">Ben Graham + CLAHE + Circular crop</div>
        </div>
        """, unsafe_allow_html=True)

    with col_cam:
        st.markdown("""
        <div class="xai-card">
            <div class="xai-title">Grad-CAM++ Attention</div>
        """, unsafe_allow_html=True)
        st.image(cam_overlay, use_container_width=True)
        st.markdown("""
            <div class="xai-caption">Deep feature-level lesion attention map</div>
        </div>
        """, unsafe_allow_html=True)

    with col_gxi:
        st.markdown("""
        <div class="xai-card">
            <div class="xai-title">Gradient × Input</div>
        """, unsafe_allow_html=True)
        st.image(grad_x_input_overlay, use_container_width=True)
        st.markdown("""
            <div class="xai-caption">Pixel-level sensitivity & feature attribution</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>📌 Research & Educational Purpose Only</div>", unsafe_allow_html=True)
