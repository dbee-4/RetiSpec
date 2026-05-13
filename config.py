import os
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "datasets_00")
SPLITS_DIR      = os.path.join(DATA_DIR, "splits")

TRAIN_CSV       = os.path.join(SPLITS_DIR, "train.csv")
VAL_CSV         = os.path.join(SPLITS_DIR, "val.csv")
TEST_CSV        = os.path.join(SPLITS_DIR, "test.csv")

OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
MODEL_DIR       = os.path.join(OUTPUT_DIR, "models")
LOG_DIR         = os.path.join(OUTPUT_DIR, "logs")
GRADCAM_DIR     = os.path.join(OUTPUT_DIR, "gradcam")
SHAP_DIR        = os.path.join(OUTPUT_DIR, "shap")

for d in [MODEL_DIR, LOG_DIR, GRADCAM_DIR, SHAP_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
NUM_CLASSES     = 5
IMAGE_SIZE      = 224
CHANNELS        = 3
NUM_WORKERS     = 4
PIN_MEMORY      = True

# ─── Training (STABILITY-OPTIMIZED) ───────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 35              # Reduced - peaks earlier
WARMUP_EPOCHS   = 3

# CRITICAL: Discriminative Learning Rates
LR              = 1e-4            # Fallback global LR
BACKBONE_LR     = 3e-5            # Low LR for pretrained backbone
HEAD_LR         = 3e-4            # Higher LR for classifier head
MIN_LR          = 1e-6

WEIGHT_DECAY    = 0.01
LABEL_SMOOTHING = 0.05            # Reduced for ordinal classification
GRAD_CLIP       = 1.0
EARLY_STOP_PAT  = 8               # Faster stopping for unstable validation

# Augmentation - Conservative for Medical Images
MIXUP_ALPHA     = 0.2             # Mild mixup
CUTMIX_ALPHA    = 0.0             # Disabled
MIXUP_PROB      = 0.35            # 35% application rate

# Loss
LOSS_FUNCTION   = "focal"
FOCAL_GAMMA     = 2.0

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "convnext_tiny"
PRETRAINED      = True
DROPOUT         = 0.15            # Reduced
STOCHASTIC_DEPTH= 0.05            # Reduced

# ─── EMA (CRITICAL FOR STABILITY) ─────────────────────────────────────────────
USE_EMA         = True
EMA_DECAY       = 0.9997

# ─── Evaluation ───────────────────────────────────────────────────────────────
PRIMARY_METRIC  = "kappa"

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model Saving ─────────────────────────────────────────────────────────────
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last_model.pth")
SEED            = 42

# ─── Advanced ─────────────────────────────────────────────────────────────────
USE_BALANCED_SAMPLER = True
SAMPLER_POWER   = 0.35            # Tempered reweighting (not sqrt=0.5)