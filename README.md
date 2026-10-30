# RetiSpec 👁️

### Explainable AI-Based Diabetic Retinopathy Detection System (FastAPI & ONNX)

## 📌 Overview

**RetiSpec** is a medical deep learning system designed for automated grading and clinical interpretability of Diabetic Retinopathy (DR) from retinal fundus images.

The model classifies fundus photographs into 5 clinical severity stages according to the International Clinical Diabetic Retinopathy Disease Severity Scale:
1. **No DR** (Grade 0)
2. **Mild** (Grade 1 - Microaneurysms only)
3. **Moderate** (Grade 2 - Microaneurysms, hemorrhages, hard exudates)
4. **Severe** (Grade 3 - 4-2-1 rule: hemorrhages, venous beading, IRMA)
5. **Proliferative DR** (Grade 4 - Neovascularization or vitreous/preretinal hemorrhage)

RetiSpec leverages **ConvNeXt-Tiny** as its core vision backbone, complemented by specialized retinal preprocessing, decision threshold optimization, Test-Time Augmentation (TTA), high-throughput **ONNX Runtime** acceleration on CPU, and dual **Explainable AI (XAI)** visualizations.

---

## 🚀 Key Features & Updates

* **FastAPI Backend:** Modern asynchronous REST API with auto-generated OpenAPI / Swagger docs at `/docs`.
* **ONNX Runtime Acceleration:** Exported `.pth` checkpoint to `.onnx` (`models/best_model.onnx`), delivering **>2x faster CPU inference** (latency reduced from ~196 ms to ~95 ms).
* **CPU-Friendly Optimization:** Built-in auto-detection for CPU environments with offline checkpoint loading (no Hugging Face runtime downloads).
* **Dual Explainable AI (XAI):**
  - **Grad-CAM++:** Generalized gradient-weighted class activation mapping targeting the final ConvNeXt block to identify macro lesion clusters.
  - **Gradient × Input:** Direct input saliency attribution ($\left| x \odot \frac{\partial y_c}{\partial x} \right|$) with Gaussian smoothing and retinal masking for fine-grained pixel sensitivity (completely replacing heavy SHAP/Numba).
* **Domain-Specific Preprocessing:** Ben Graham's color constancy method, circular ROI cropping, aspect-ratio-preserving padding, and CLAHE contrast enhancement.
* **Docker & Docker Compose:** Containerized CPU-optimized deployment in a single command.
* **Interactive Web Interface:** Glassmorphism UI with real-time fundus preview, probability distribution bars, and diagnostic reports.

---

## 📁 File Structure & Concepts Used

| File | Primary Role | Core Concepts & Techniques Used |
| :--- | :--- | :--- |
| [`preprocessing.py`](file:///E:/RetiSpec/preprocessing.py) | Retinal Image Pipeline | **Ben Graham's Technique** (unsharp masking via local average subtraction), **CLAHE** (Contrast Limited Adaptive Histogram Equalization in LAB space), **Circular ROI Cropping** (black border removal), **Aspect-Ratio-Preserving Padding**, ImageNet normalization, Albumentations Compose pipeline. |
| [`model.py`](file:///E:/RetiSpec/model.py) | Architecture & Weights | **ConvNeXt-Tiny** backbone ($7\times 7$ depthwise convolutions, inverted bottleneck, LayerNorm in channels-last, GELU activations, Stochastic Depth / drop-path rate), Global Average Pooling, custom 2-layer classification head ($768 \rightarrow 256 \rightarrow 5$) with **Truncated Normal** weight initialization. |
| [`config.py`](file:///E:/RetiSpec/config.py) | Central Configuration | Hardware auto-detection (`cpu` vs `cuda`), path resolution with fallback, discriminative learning rates, class sampler exponent, EMA decay rate, early stopping patience. |
| [`tta.py`](file:///E:/RetiSpec/tta.py) | Test-Time Augmentation | **Ensemble TTA**: 6 anatomically safe perturbations (identity, horizontal flip, $\pm 5^\circ$ rotations, $\pm 5\%$ brightness jitter; vertical flips avoided to preserve optic disc orientation), soft-voting probability averaging, CPU/CUDA dynamic autocast. |
| [`threshold_optimization.py`](file:///E:/RetiSpec/threshold_optimization.py) | Threshold Optimization | **Ordinal Classification Modeling** (modeling progressive DR severity stages), cumulative probability thresholding, **Quadratic Weighted Kappa** optimization via grid search, class-specific F1 threshold tuning. |
| [`train.py`](file:///E:/RetiSpec/train.py) | Model Training | **Focal Loss** ($\gamma=2.0$) with class-frequency weighting, **Exponential Moving Average (EMA)** of weights ($\beta=0.9997$), **Mixup Augmentation** with linear decay, **Discriminative Learning Rates** (backbone $3\times 10^{-5}$ vs head $3\times 10^{-4}$), AdamW optimizer, gradient clipping. |
| [`dataset.py`](file:///E:/RetiSpec/dataset.py) | Data Ingestion | PyTorch `Dataset` & `DataLoader`, **WeightedRandomSampler** with tempered power reweighting ($\text{power}=0.35$ to prevent over-sampling rare classes), domain-safe data augmentations, fail-fast image reading. |
| [`evaluate.py`](file:///E:/RetiSpec/evaluate.py) | Multi-Stage Evaluation | Baseline vs TTA vs Threshold-optimized comparison, Quadratic Weighted Kappa metric, Confusion Matrix (raw counts and normalized recall), classification report. |
| [`xai_gradcam.py`](file:///E:/RetiSpec/xai_gradcam.py) | Coarse Lesion Attention | **Grad-CAM++** (higher-order positive partial derivative weighting for multiple lesion focus), activation hook on final ConvNeXt block, JET colormap blending, circular background suppression. |
| [`xai_gradientxinput.py`](file:///E:/RetiSpec/xai_gradientxinput.py) | Fine Pixel Saliency | **Gradient × Input** feature attribution ($\left\| x \odot \frac{\partial y_c}{\partial x} \right\|$), channel-wise sensitivity aggregation, Gaussian kernel smoothing, retinal background masking. |
| [`export_onnx.py`](file:///E:/RetiSpec/export_onnx.py) | ONNX Export & Benchmark | PyTorch to ONNX Opset 14 export, constant folding, dynamic batch dimension, ONNX model validation, **ONNX Runtime CPU acceleration** benchmark vs PyTorch CPU. |
| [`app.py`](file:///E:/RetiSpec/app.py) | Web API & Server | **FastAPI** application, async multipart form image ingestion, model inference (PyTorch CPU / ONNX Runtime), real-time Grad-CAM++ and Gradient × Input generation, base64 response serialization, Jinja2 template rendering. |
| [`templates/index.html`](file:///E:/RetiSpec/templates/index.html) & [`static/`](file:///E:/RetiSpec/static/) | Frontend Interface | Drag-and-drop file upload, instant image preview, asynchronous JSON API communication, animated class probability bars, responsive side-by-side diagnostic visualization grid. |

---

## 🛠️ Quickstart (Local Virtual Environment)

### 1. Activate Virtual Environment
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 2. Export / Benchmark ONNX (Optional - already pre-generated)
```bash
python export_onnx.py
```

### 3. Run FastAPI Web Server
```bash
python app.py
```
Open your browser at:
* **Web UI:** [http://localhost:8000](http://localhost:8000)
* **Interactive API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 🐳 Running with Docker

### Option A: Using Docker Compose (Recommended)
```bash
docker compose up --build
```

### Option B: Using Docker CLI
```bash
# Build the CPU-optimized container
docker build -t retispec:latest .

# Run container on port 8000
docker run -d -p 8000:8000 --name retispec_app retispec:latest
```
Access the application at `http://localhost:8000`.

---

## 🔬 Explainable AI (XAI) Methods

1. **Grad-CAM++ (Attention Map):** Identifies macro spatial regions and feature activations within the ConvNeXt deep feature maps that triggered the class prediction.
2. **Gradient × Input (Pixel Attribution):** Calculates element-wise sensitivity by multiplying input pixel intensities by backpropagated gradients w.r.t the predicted class score, highlighting specific microaneurysms, hemorrhages, and exudates.
