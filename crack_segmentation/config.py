"""
config.py - Configuration file for Crack Weakly Supervised Segmentation

Dataset: 4032x3024 resolution crack images with ground truth masks
Framework: IRNet (Ahn et al., CVPR 2019) adapted for crack detection
"""

import torch

# ============================================================
# PATHS (sesuaikan dengan struktur folder Anda)
# ============================================================
IMG_DIR = "data/images"              # Folder berisi gambar crack
MASK_DIR = "data/masks"              # Folder berisi ground truth mask
OUTPUT_DIR = "outputs"               # Folder untuk menyimpan hasil

# ============================================================
# DATASET CONFIGURATION
# ============================================================
# Original image resolution
IMG_HEIGHT = 4032
IMG_WIDTH = 3024

# Patch extraction untuk training
PATCH_SIZE = 512                     # Ukuran patch untuk training
PATCH_STRIDE = 256                   # Stride untuk sliding window
MIN_CRACK_RATIO = 0.03              # Minimum ratio crack pixels untuk patch positif
MAX_NEG_RATIO = 2.0                 # Ratio negative:positive patches

# Untuk inference pada full image
INFERENCE_PATCH_SIZE = 512
INFERENCE_STRIDE = 256              # Overlap untuk smooth stitching

# ============================================================
# MODEL CONFIGURATION
# ============================================================
NUM_CLASSES = 2                     # 0: background, 1: crack

# Feature map size (input / 4 karena backbone stride)
FEAT_H = PATCH_SIZE // 4            # 128 untuk patch 512x512
FEAT_W = PATCH_SIZE // 4            # 128 untuk patch 512x512

# IRNet PathIndex parameters
RADIUS = 5                          # Search radius untuk neighbor pairs

# ============================================================
# TRAINING CONFIGURATION
# ============================================================
# Stage 1: Classification Network (CAM)
CAM_EPOCHS = 10                      # Increased from 2 to 30 for better CAM learning
CAM_BATCH_SIZE = 8
CAM_LR = 1e-4                        # Reverted to original - 5e-4 was too high
CAM_WEIGHT_DECAY = 1e-5

# Focal Loss parameters
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Stage 2+3: IRNet Training
IRN_EPOCHS = 5                      # Increased from 20
IRN_BATCH_SIZE = 4                  # Lebih kecil karena lebih memory-intensive
IRN_LR = 1e-3                        # Keep at 1e-3 for fast learning
IRN_WEIGHT_DECAY = 1e-5

# Loss weights (from IRNet paper)
GAMMA = 2.0                          # Increased from 0.5 - displacement needs more weight
LAMBDA_POS = 1.0                    # Weight untuk positive affinity loss
LAMBDA_NEG = 1.0                    # Weight untuk negative affinity loss

# ============================================================
# PSEUDO LABEL SYNTHESIS
# ============================================================
# Random walk parameters
BETA = 8                            # β untuk affinity power (Eq 12)
T_WALK = 256                        # Random walk steps
BG_QUANTILE = 0.25                  # Background threshold quantile
B_PENALTY = 0.7                     # Boundary penalty untuk seed (1-boundary_prob)

# Instance segmentation
CC_CONNECTIVITY = 8                 # Connectivity untuk connected components
MIN_INSTANCE_SIZE = 50              # Minimum pixel untuk instance yang valid

# Displacement refinement
REFINE_ITERATIONS = 100             # Iterasi untuk mean-shift refinement
REFINE_RADIUS = 1.0                 # Radius untuk neighbor averaging

# DenseCRF post-processing
USE_DENSECRF = True                 # Enable DenseCRF refinement
DCRF_ITER = 6                      # Gentle refinement
DCRF_POS_W = 10                     # Light spatial smoothing
DCRF_POS_XY_STD = 3                # Small spatial kernel
DCRF_BI_W = 100                      # Light bilateral weight
DCRF_BI_XY_STD = 10                 # Moderate bilateral spatial extent
DCRF_BI_RGB_STD = 3                # Good color tolerance

# Hybrid CAM + IRN parameters
USE_BOUNDARY_REFINEMENT = True      # Use IRN boundary map to refine CAM
BOUNDARY_SUPPRESSION_WEIGHT = 0.5   # How much to suppress CAM at boundaries (0-1)
USE_DISPLACEMENT_CLUSTERING = False  # Use displacement field for instance separation
DISPLACEMENT_CLUSTER_THRESHOLD = 5.0 # Distance threshold for clustering

# ============================================================
# DATA AUGMENTATION
# ============================================================
# Training augmentation
AUG_H_FLIP_PROB = 0.5
AUG_V_FLIP_PROB = 0.5
AUG_ROTATION_DEGREES = 15
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2

# ImageNet normalization (standard untuk ResNet)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# ============================================================
# DEVICE & OPTIMIZATION
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4                     # DataLoader workers
PIN_MEMORY = True                   # Faster CPU-GPU transfer

# Mixed precision training (untuk memory efficiency)
USE_AMP = True                      # Automatic Mixed Precision

# ============================================================
# EVALUATION & VISUALIZATION
# ============================================================
# Threshold untuk binary prediction
BINARY_THRESHOLD = 0.5

# Visualization
COLORMAP = 'jet'                    # Colormap untuk CAM visualization
SAVE_VISUALIZATIONS = True          # Save intermediate visualizations

# Metrics
COMPUTE_METRICS = True              # Compute IoU, Precision, Recall, F1
