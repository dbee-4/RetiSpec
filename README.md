# RetiSpec 👁️

### Explainable AI-Based Diabetic Retinopathy Detection System

## 📌 Overview

RetiSpec is an AI/ML project developed for detecting and classifying Diabetic Retinopathy (DR) from retinal fundus images using deep learning and Explainable AI techniques.

The model classifies retinal images into 5 severity stages:

* No DR
* Mild
* Moderate
* Severe
* Proliferative DR

The project uses **ConvNeXt-Tiny** as the backbone architecture along with preprocessing, threshold optimization, Test-Time Augmentation (TTA), and Explainable AI visualizations.

---

# 📂 Datasets Used

The model was trained using multiple publicly available retinal fundus datasets:

* **APTOS 2019 Blindness Detection**
* **Messidor-2**
* **EyePACS**

These datasets were combined to improve model generalization and robustness across different retinal image distributions.

---

# 🚀 Features

* Deep learning-based DR classification
* ConvNeXt-Tiny architecture
* Ben Graham retinal preprocessing
* Test-Time Augmentation (TTA)
* Threshold optimization
* Explainable AI visualizations
* Flask web application
* Interactive frontend using HTML, CSS, and JavaScript

---

# 📊 Performance

* **Accuracy:** 93%
* **Quadratic Weighted Kappa Score:** 0.81

---

# 🔥 Explainable AI (XAI)

The project includes Explainable AI techniques to improve interpretability and transparency of predictions.

### Techniques Used

* Grad-CAM++
* Gradient × Input visualization

These visualizations help identify important retinal regions influencing the model prediction.

---

# 🛠️ Tech Stack

## Backend

* Python
* Flask
* PyTorch

## Frontend

* HTML
* CSS
* JavaScript

## Libraries

* OpenCV
* Albumentations
* timm
* torchvision
* NumPy
* Pillow

---

# 📁 Project Structure

```text id="0i4vd0"
RetiSpec/
│
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│   └── index.html
│
├── outputs/
│   └── models/
│       └── optimized_thresholds.pkl
│
├── app.py
├── config.py
├── dataset.py
├── evaluate.py
├── model.py
├── preprocessing.py
├── requirements.txt
├── threshold_optimization.py
├── train.py
├── tta.py
├── utils.py
├── xai_gradcam.py
├── xai_shap.py
```

---

# ▶️ Run the Project

Install dependencies:

```bash id="y6wy6r"
pip install -r requirements.txt
```

Run the Flask server:

```bash id="jzlls7"
python app.py
```

---

# 📸 Outputs

The repository includes:

* prediction outputs
* Grad-CAM visualizations
* frontend screenshots

---

# 👩‍💻 Developed By

* Dena D
* Jeffisha Jemi J

Final Year AI/ML Engineering Project

---

# ⚠️ Note

This project is developed for educational and research purposes only.
