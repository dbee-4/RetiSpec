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
# Check if models exist in base directory 'models/' first, otherwise fallback to 'outputs/models/'
if os.path.exists(os.path.join(BASE_DIR, "models", "best_model.pth")):
    MODEL_DIR   = os.path.join(BASE_DIR, "models")
else:
    MODEL_DIR   = os.path.join(OUTPUT_DIR, "models")

LOG_DIR         = os.path.join(OUTPUT_DIR, "logs")
GRADCAM_DIR     = os.path.join(OUTPUT_DIR, "gradcam")
GRADXINPUT_DIR  = os.path.join(OUTPUT_DIR, "gradient_x_input")

for d in [MODEL_DIR, LOG_DIR, GRADCAM_DIR, GRADXINPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
NUM_CLASSES     = 5
IMAGE_SIZE      = 224
CHANNELS        = 3
NUM_WORKERS     = 0 if os.name == 'nt' else 4   # 0 workers on Windows avoids multiprocessing lockups
PIN_MEMORY      = torch.cuda.is_available()

# ─── Training (STABILITY-OPTIMIZED) ───────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 35
WARMUP_EPOCHS   = 3

# Discriminative Learning Rates
LR              = 1e-4
BACKBONE_LR     = 3e-5
HEAD_LR         = 3e-4
MIN_LR          = 1e-6

WEIGHT_DECAY    = 0.01
LABEL_SMOOTHING = 0.05
GRAD_CLIP       = 1.0
EARLY_STOP_PAT  = 8

# Augmentation - Conservative for Medical Images
MIXUP_ALPHA     = 0.2
CUTMIX_ALPHA    = 0.0
MIXUP_PROB      = 0.35

# Loss
LOSS_FUNCTION   = "focal"
FOCAL_GAMMA     = 2.0

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "convnext_tiny"
PRETRAINED      = True
DROPOUT         = 0.15
STOCHASTIC_DEPTH= 0.05

# ─── EMA (CRITICAL FOR STABILITY) ─────────────────────────────────────────────
USE_EMA         = True
EMA_DECAY       = 0.9997

# ─── Evaluation ───────────────────────────────────────────────────────────────
PRIMARY_METRIC  = "kappa"

# ─── Device ───────────────────────────────────────────────────────────────────
# Automatically select CUDA if available; seamlessly fallback to CPU
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model Paths ──────────────────────────────────────────────────────────────
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last_model.pth")
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.onnx")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "optimized_thresholds.pkl")
SEED            = 42

# ─── Advanced ─────────────────────────────────────────────────────────────────
USE_BALANCED_SAMPLER = True
SAMPLER_POWER   = 0.35
