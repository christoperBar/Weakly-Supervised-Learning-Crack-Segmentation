"""
config.py - Configuration file for Crack Weakly Supervised Segmentation

Dataset: 4032x3024 resolution crack images with ground truth masks
Framework: IRNet (Ahn et al., CVPR 2019) adapted for crack detection
"""

import os
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
MAX_NEG_RATIO = 1.0                 # Ratio negative:positive patches (1:1)

# Train/validation/test split (berbasis citra, bukan patch)
N_TRAIN_IMAGES = 150                # Jumlah citra untuk train
N_VAL_IMAGES = 59                   # Jumlah citra untuk validation
N_TEST_IMAGES = 50                  # Jumlah citra untuk test
RANDOM_SEED = 42                    # Seed untuk reproducible split

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
CAM_EPOCHS = 50                      # Increased from 2 to 30 for better CAM learning
CAM_BATCH_SIZE = 8
CAM_LR = 0.1                        # Safer SGD LR for stable fine-tuning
CAM_WEIGHT_DECAY = 0.0               # Disable weight decay

# Focal Loss parameters
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Stage 2+3: IRNet Training
IRN_EPOCHS = 50                      # Increased from 20
IRN_BATCH_SIZE = 8                  
IRN_LR = 6e-5                     
IRN_WEIGHT_DECAY = 6e-6 

# Early stopping
EARLY_STOPPING_ENABLED = True
CAM_EARLY_STOP_PATIENCE = 8          # Stop jika metric CAM tidak membaik selama N epoch
CAM_EARLY_STOP_MIN_DELTA = 1e-4      # Minimal kenaikan metric agar dianggap membaik
IRN_EARLY_STOP_PATIENCE = 10         # Stop jika loss IRNet tidak membaik selama N epoch
IRN_EARLY_STOP_MIN_DELTA = 1e-4      # Minimal penurunan loss agar dianggap membaik

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
DCRF_ITER = 10                      # Gentle refinement
DCRF_POS_W = 3                     # Light spatial smoothing
DCRF_POS_XY_STD = 3                # Small spatial kernel
DCRF_BI_W = 10                      # Light bilateral weight
DCRF_BI_XY_STD = 150                 # Moderate bilateral spatial extent
DCRF_BI_RGB_STD = 30                # Good color tolerance

# DCRF_ITER = 20                      # Gentle refinement
# DCRF_POS_W = 10                     # Light spatial smoothing
# DCRF_POS_XY_STD = 3                # Small spatial kernel
# DCRF_BI_W = 150                      # Light bilateral weight
# DCRF_BI_XY_STD = 10                 # Moderate bilateral spatial extent
# DCRF_BI_RGB_STD = 3                 # Good color tolerance


# Hybrid CAM + IRN parameters
USE_BOUNDARY_REFINEMENT = True      # Use IRN boundary map to refine CAM
BOUNDARY_SUPPRESSION_WEIGHT = 0.15  # How much to suppress CAM at boundaries (0-1)
USE_DISPLACEMENT_CLUSTERING = False  # Use displacement field for instance separation
DISPLACEMENT_CLUSTER_THRESHOLD = 5.0 # Distance threshold for clustering

# Restorative fusion: IRN should help recover thin/disconnected crack regions
IRN_RESTORE_GAIN = 0.75             # Strength of IRN support injection into CAM
IRN_RESTORE_SEED_THRESH = 0.38      # High-confidence CAM seed threshold
IRN_RESTORE_DILATION = 6            # Neighborhood around seeds for safe IRN support
IRN_RESTORE_SUPPORT_THRESH = 0.20   # Minimum IRN support to be considered recoverable
IRN_RESTORE_LOW_CONF_THRESH = 0.55  # Restore mainly where CAM confidence is still low
IRN_RESTORE_CONNECT_DILATION = 6    # Connectivity anchor dilation around CAM seeds

# ============================================================
# DATA AUGMENTATION
# ============================================================
# Training augmentation
AUG_H_FLIP_PROB = 0.5
AUG_V_FLIP_PROB = 0.5
AUG_ROTATION_DEGREES = 15
AUG_BRIGHTNESS = 0.0
AUG_CONTRAST = 0.0

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

# ============================================================
# STAGE 5: RESNET50-UNET TRAINING (FROM PSEUDO LABELS)
# ============================================================
# Stage 5 uses original images + pseudo labels generated in Stage 4
STAGE5_IMAGE_DIR = IMG_DIR
STAGE5_LABEL_DIR = os.path.join(OUTPUT_DIR, 'pseudo_labels')
STAGE5_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'stage5_unet')
STAGE5_TEST_IMAGE_DIR = 'data/images'
STAGE5_TEST_MASK_DIR = 'data/masks'

# Training params
STAGE5_EPOCHS = 30
STAGE5_BATCH_SIZE = 8
STAGE5_LR = 0.1
STAGE5_PATCH_SIZE = 224
STAGE5_PATCHES_PER_IMG = 20
