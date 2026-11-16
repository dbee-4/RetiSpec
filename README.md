# RetiSpec 👁️
### Explainable AI-Based Diabetic Retinopathy Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch%20CPU-EE4C2C.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%20(<100ms)-005CED.svg)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ⚡ 30-Second Summary for Recruiters & Reviewers

**RetiSpec** is a complete, production-grade medical computer vision system that classifies retinal fundus photographs into 5 diabetic retinopathy severity stages while providing clinicians with dual, real-time Explainable AI (XAI) attribution maps.

* **Clinical Goal:** Automate detection of Diabetic Retinopathy (DR)—the leading cause of preventable adult blindness—grading images into **No DR**, **Mild**, **Moderate**, **Severe**, or **Proliferative DR**.
* **Vision Backbone:** **ConvNeXt-Tiny** fine-tuned with domain-specific retinal preprocessing (Ben Graham's color constancy, circular ROI cropping, and LAB-space CLAHE).
* **Dual Explainable AI (XAI):** Solves the medical "black box" dilemma by generating two distinct visual interpretations for every diagnosis:
  1. **Grad-CAM++:** Highlights macro lesion clusters and diseased regions across deep feature maps.
  2. **Gradient × Input:** Fine-grained pixel-level attribution mapping subtle microaneurysms, cotton wool spots, and hemorrhages.
* **Production Engineering:**
  * **Interactive Streamlit Web Dashboard:** Clean, intuitive UI with built-in sample selector, real-time prediction distribution, and side-by-side heatmaps.
  * **FastAPI Backend:** Asynchronous REST API serving web UI and OpenAPI docs (`/docs`).
  * **ONNX Runtime Acceleration:** Pre-converted ONNX model (`models/best_model.onnx`) cuts CPU inference latency from **~196 ms down to ~95 ms** (>2× speedup).
  * **Automated TTA:** 6-fold Test-Time Augmentation (safe horizontal flips, micro-rotations, and brightness jitter) runs on every image for robust, invariant predictions.
  * **Zero Cloud GPU Requirement:** Fully optimized to run on standard CPU hardware.

---

## 🔬 Diagnostic Visual Pipeline

When a retinal fundus image is uploaded, RetiSpec executes a 3-step diagnostic pipeline:

```text
Fundus Image ──► Preprocessing ──► ConvNeXt-Tiny + TTA ──► Dual Explainable AI ──► Clinical Report
                 (Ben Graham +      (PyTorch / ONNX)       (Grad-CAM++ &           (Class Probs &
                  Circular Crop)                            Gradient × Input)       Confidence)
```

| Visual Output | Method | Clinical Purpose |
| :--- | :--- | :--- |
| **Preprocessed Input** | Ben Graham + CLAHE + Circular Crop | Standardizes illumination, removes camera artifacts, and isolates retinal tissue. |
| **Grad-CAM++ Attention** | Higher-order feature map activation | Identifies broad spatial regions and vessel arches driving the classification. |
| **Gradient × Input** | $\left\| x \odot \frac{\partial y_c}{\partial x} \right\|$ Sensitivity | Pinpoints exact micro-lesions, hemorrhages, and exudate boundaries at the pixel level. |

---

## 📁 Project Structure & Core Concepts

| File | Role | Techniques & Concepts Used |
| :--- | :--- | :--- |
| [`streamlit_app.py`](file:///e:/RetiSpec/streamlit_app.py) | Cloud & Local Frontend | Streamlit application, sample selector, cached inference engines, reactive diagnostic report cards. |
| [`app.py`](file:///e:/RetiSpec/app.py) | REST API Server | FastAPI async engine, multipart image ingestion, dual XAI generation, dynamic CPU thread limiting. |
| [`model.py`](file:///e:/RetiSpec/model.py) | Deep Learning Model | ConvNeXt-Tiny backbone, $7\times 7$ depthwise convolutions, inverted bottleneck, 2-layer classifier head. |
| [`preprocessing.py`](file:///e:/RetiSpec/preprocessing.py) | Retinal Pipeline | **Ben Graham's method** (local average subtraction), **CLAHE** in LAB space, circular ROI auto-cropping, aspect padding. |
| [`xai_gradcam.py`](file:///e:/RetiSpec/xai_gradcam.py) | Macro Explainability | **Grad-CAM++** with positive partial derivative weighting on the final ConvNeXt stage. |
| [`xai_gradientxinput.py`](file:///e:/RetiSpec/xai_gradientxinput.py) | Pixel Explainability | **Gradient × Input** feature saliency, channel aggregation, Gaussian smoothing, retinal tissue masking. |
| [`export_onnx.py`](file:///e:/RetiSpec/export_onnx.py) | Runtime Optimization | PyTorch to ONNX Opset 14 export, dynamic batching, CPU throughput benchmark. |
| [`tta.py`](file:///e:/RetiSpec/tta.py) | Test-Time Augmentation | 6-fold anatomically safe ensemble transforms with soft-voting probability averaging. |
| [`threshold_optimization.py`](file:///e:/RetiSpec/threshold_optimization.py) | Post-Processing | Ordinal classification modeling, Quadratic Weighted Kappa (QWK) grid optimization. |
| [`train.py`](file:///e:/RetiSpec/train.py) & [`evaluate.py`](file:///e:/RetiSpec/evaluate.py) | Training & Validation | Focal Loss ($\gamma=2.0$), Exponential Moving Average (EMA), discriminative learning rates, QWK evaluation. |
| [`templates/`](file:///e:/RetiSpec/templates/) & [`static/`](file:///e:/RetiSpec/static/) | Classic Web Interface | Dark glassmorphism UI, drag-and-drop image upload, real-time probability visualization bars. |

---

## 🛠️ Quickstart (Run Locally)

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/dbee-4/RetiSpec.git
cd RetiSpec

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Launch the Web Application

#### Option A: Streamlit Interface (Recommended)
```bash
streamlit run streamlit_app.py
```
* **URL:** [http://localhost:8501](http://localhost:8501)

#### Option B: FastAPI Backend & Glassmorphism UI
```bash
python app.py
```
* **Web UI:** [http://localhost:8000](http://localhost:8000)
* **Interactive API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📌 Research & Educational Purpose Only
