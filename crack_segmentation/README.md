# Crack Weakly Supervised Segmentation

Implementasi **weakly supervised semantic segmentation** untuk deteksi crack pada gambar resolusi tinggi (4032x3024) menggunakan framework **IRNet** (Ahn et al., CVPR 2019).

## 📋 Overview

Project ini mengimplementasikan pipeline end-to-end untuk crack segmentation dengan hanya menggunakan **image-level labels** (crack/no-crack) sebagai supervision, tanpa memerlukan pixel-level annotations yang mahal.

### Pipeline

```
Stage 1: Classification Network (CAM)
    ├─ Train ResNet50-based classifier
    └─ Extract Class Activation Maps (CAMs)
          ↓
Stage 2: Mine Inter-Pixel Relations
    ├─ Extract CAMs dari training images
    └─ Generate positive/negative pixel pairs
          ↓
Stage 3: Train IRNet
    ├─ Edge Branch → Boundary detection
    └─ Displacement Branch → Instance field
          ↓
Stage 4: Pseudo Label Synthesis
    ├─ Extract CAM + Edge + Displacement
    ├─ Random walk propagation
    └─ Generate pseudo instance masks
```

## 🎯 Features

- ✅ Support untuk gambar resolusi tinggi (4032x3024)
- ✅ Patch-based training dengan sliding window
- ✅ Automatic mixed precision (AMP) untuk memory efficiency
- ✅ Test-time augmentation (TTA) untuk robust predictions
- ✅ Visualization tools untuk setiap stage
- ✅ Evaluation metrics (IoU, Precision, Recall, F1)
- ✅ Modular code structure untuk easy customization

## 📁 Project Structure

```
crack_segmentation/
├── config.py                 # Konfigurasi dataset dan training
├── dataset.py               # Dataset classes dan data loading
├── path_index.py            # PathIndex untuk IRNet (dari referensi)
├── resnet50.py              # ResNet50 backbone (dari referensi)
├── resnet50_cam.py          # CAM network (dari referensi)
├── resnet50_irn.py          # IRNet model (dari referensi)
├── train_stage1.py          # Training script Stage 1
├── train_stage2_3.py        # Training script Stage 2+3
├── inference.py             # Inference script Stage 4
├── main.py                  # Main pipeline script
├── README.md                # Dokumentasi ini
│
├── data/                    # Data directory
│   ├── images/              # Crack images (4032x3024)
│   └── masks/               # Ground truth masks
│
└── outputs/                 # Output directory
    ├── cam_net_best.pth     # Best CAM model
    ├── irnet_best.pth       # Best IRNet model
    ├── pseudo_labels/       # Generated pseudo labels
    └── visualizations/      # Visualization outputs
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install torch torchvision opencv-python numpy scipy matplotlib tqdm

# Install DenseCRF (optional, for post-processing)
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

### 2. Prepare Dataset

Struktur folder dataset:

```
data/
├── images/
│   ├── image1.jpg    # 4032x3024
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1.png    # Ground truth (same size)
    ├── image2.png
    └── ...
```

**Catatan**: Masks harus binary (0=background, 255=crack).

### 3. Configure Settings

Edit `config.py` untuk menyesuaikan path dan parameters:

```python
# Dataset paths
IMG_DIR = "data/images"
MASK_DIR = "data/masks"
OUTPUT_DIR = "outputs"

# Training parameters
CAM_EPOCHS = 25
IRN_EPOCHS = 20
PATCH_SIZE = 512
PATCH_STRIDE = 256
```

### 4. Run Pipeline

#### Option A: Run Full Pipeline

```bash
python main.py --stage all
```

#### Option B: Run Stage by Stage

```bash
# Stage 1: Train CAM network
python main.py --stage 1

# Stage 2+3: Train IRNet
python main.py --stage 2

# Stage 4: Generate pseudo labels
python main.py --stage 3
```

#### Option C: Inference Only

```bash
# Jika model sudah trained
python main.py --inference
```

## 📊 Training Details

### Stage 1: Classification Network

- **Model**: ResNet50 + CAM
- **Loss**: Focal Loss (α=0.25, γ=2.0)
- **Optimizer**: Adam (lr=1e-4)
- **Batch Size**: 8
- **Epochs**: 25
- **Data Augmentation**:
  - Random horizontal/vertical flip
  - Random rotation (±15°)
  - Color jitter (brightness, contrast)

**Output**: `cam_net_best.pth`

### Stage 2+3: IRNet Training

- **Model**: IRNet (Edge + Displacement branches)
- **Loss**: Affinity + Displacement losses
- **Optimizer**: Adam (lr=1e-4)
- **Batch Size**: 4
- **Epochs**: 20
- **γ (displacement weight)**: 5.0

**Output**: `irnet_best.pth`

### Stage 4: Pseudo Label Generation

- **Random Walk**: 256 steps
- **β (affinity power)**: 8
- **Background quantile**: 0.25
- **Boundary penalty**: 0.7

**Output**: Binary segmentation masks

## 🎨 Visualization

Setiap stage menghasilkan visualizations:

**Stage 1**: Training curves (loss, accuracy)
**Stage 2+3**: Loss decomposition (affinity, displacement)
**Stage 4**: 
- Original image
- CAM heatmap
- Boundary map
- Displacement field
- Pseudo label
- Overlay

Visualizations disimpan di `outputs/visualizations/`

## 📈 Evaluation Metrics

Jika ground truth masks tersedia:

- **IoU** (Intersection over Union)
- **Precision**
- **Recall**
- **F1-Score**

Metrics dihitung per-image dan averaged across dataset.

## ⚙️ Configuration Options

### Dataset Configuration

```python
# Image size
IMG_HEIGHT = 4032
IMG_WIDTH = 3024

# Patch extraction
PATCH_SIZE = 512          # Patch size untuk training
PATCH_STRIDE = 256        # Overlap = PATCH_SIZE - PATCH_STRIDE
MIN_CRACK_RATIO = 0.03    # Min crack ratio untuk positive patch

# Inference
INFERENCE_PATCH_SIZE = 512
INFERENCE_STRIDE = 256    # Lebih kecil = lebih smooth tapi lebih lambat
```

### Training Configuration

```python
# CAM Network
CAM_EPOCHS = 25
CAM_BATCH_SIZE = 8
CAM_LR = 1e-4

# IRNet
IRN_EPOCHS = 20
IRN_BATCH_SIZE = 4        # Reduce jika OOM
IRN_LR = 1e-4
GAMMA = 5.0               # Displacement loss weight

# Device
DEVICE = "cuda"           # atau "cpu"
USE_AMP = True            # Mixed precision training
```

### Pseudo Label Configuration

```python
# Random walk
BETA = 8                  # Affinity power
T_WALK = 256              # Random walk iterations
BG_QUANTILE = 0.25        # Background threshold

# Instance segmentation
CC_CONNECTIVITY = 8       # 4 atau 8
MIN_INSTANCE_SIZE = 50    # Minimum pixels per instance
```

## 🔧 Troubleshooting

### Out of Memory (OOM)

**Solutions**:
1. Reduce `PATCH_SIZE` (512 → 384 atau 256)
2. Reduce `BATCH_SIZE` (8 → 4 atau 2)
3. Enable `USE_AMP = True` (mixed precision)
4. Reduce `NUM_WORKERS` in DataLoader

### Training Too Slow

**Solutions**:
1. Increase `PATCH_STRIDE` (256 → 384) - fewer patches
2. Reduce `IRN_EPOCHS` (20 → 15)
3. Use GPU if available
4. Reduce `T_WALK` (256 → 128) untuk Stage 4

### Poor Results

**Solutions**:
1. Increase `CAM_EPOCHS` (25 → 40)
2. Tune `MIN_CRACK_RATIO` berdasarkan dataset
3. Adjust `BETA` dan `BG_QUANTILE` untuk better random walk
4. Check data quality dan class balance

## 📝 Implementation Notes

### Perbedaan dengan Original IRNet

1. **Dataset**: Original menggunakan PASCAL VOC (multi-class), implementasi ini untuk binary crack detection
2. **Resolution**: Original 320x320 patches, implementasi ini support 4032x3024 full images
3. **Sliding Window**: Implementasi patch-based processing untuk handle large images
4. **Simplifikasi**: Inter-pixel relation mining disederhanakan untuk efficiency

### Memory Management

Untuk images 4032x3024:
- **Training**: Patches 512x512, batch size 4-8
- **Inference**: Sliding window dengan overlap, feature maps di-downsample untuk random walk
- **Peak memory**: ~8-12GB GPU untuk batch size 8

### Speed Optimization

- Mixed precision training (AMP)
- Efficient sliding window dengan numpy vectorization
- Sparse matrix untuk random walk (scipy.sparse)
- Pre-computed PathIndex untuk displacement loss

## 📚 References

1. **IRNet Paper**: Ahn, J., & Kwak, S. (2018). "Learning pixel-level semantic affinity with image-level supervision for weakly supervised semantic segmentation." CVPR 2019.

2. **CAM**: Zhou, B., et al. (2016). "Learning deep features for discriminative localization." CVPR 2016.

3. **Random Walk**: Grady, L. (2006). "Random walks for image segmentation." TPAMI 2006.

## 📄 License

Kode ini dibuat untuk tujuan penelitian dan edukasi. Silakan gunakan dan modifikasi sesuai kebutuhan.

## 🙏 Acknowledgments

- Original IRNet implementation oleh Jiwoon Ahn
- ResNet50 pretrained weights dari PyTorch Model Zoo
- DeepCrack dataset untuk referensi

## 📧 Contact

Jika ada pertanyaan atau issues, silakan buat issue di repository ini atau hubungi developer.

---

**Happy Coding!** 🚀
