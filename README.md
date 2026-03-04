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

## 📁 Project Structure

```
├── crack_segmentation/
│   ├── config.py                # Konfigurasi dataset dan training
│   ├── dataset.py               # Dataset classes dan data loading
│   ├── path_index.py            # PathIndex untuk IRNet (dari referensi)
│   ├── resnet50.py              # ResNet50 backbone (dari referensi)
│   ├── resnet50_cam.py          # CAM network (dari referensi)
│   ├── resnet50_irn.py          # IRNet model (dari referensi)
│   ├── train_stage1.py          # Training script Stage 1
│   ├── train_stage2_3.py        # Training script Stage 2+3
│   ├── inference.py             # Inference script Stage 4
│   ├── main.py                  # Main pipeline script            
│   │
│   ├── data/                    # Data directory
│   │   ├── images/              # Crack images (4032x3024)
│   │   └── masks/               # Ground truth masks
│   │
│   └── outputs/                 # Output directory
│       ├── cam_net_best.pth     # Best CAM model
│       ├── irnet_best.pth       # Best IRNet model
│       ├── pseudo_labels/       # Generated pseudo labels
│       └── visualizations/      # Visualization outputs
├── Dockerfile
├── README.md
├── .gitignore
```

## 🧪 Test Run

### Quick Test

```bash
# Test semua components
python quick_test.py

# Run example usage
python example_usage.py
```

## 🚀 Quick Start

### 1. Run Docker

```bash
docker run --gpus all -it \
  --shm-size=8g \
  -v "$(pwd -W)/crack_segmentation:/workspace/crack_segmentation" \
  -v "$(pwd -W)/data:/workspace/data" \
  crack-irn:py310
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

```

#### Option C: Inference Only

```bash
# Jika model sudah trained
python main.py --inference
```

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

## 📝 Implementation Notes

### Perbedaan dengan Original IRNet

1. **Dataset**: Original menggunakan PASCAL VOC (multi-class), implementasi ini untuk binary crack detection
2. **Resolution**: Original 320x320 patches, implementasi ini support 4032x3024 full images
3. **Sliding Window**: Implementasi patch-based processing untuk handle large images
4. **Simplifikasi**: Inter-pixel relation mining disederhanakan untuk efficiency


## 📚 References

1. **IRNet Paper**: Ahn, J., & Kwak, S. (2018). "Learning pixel-level semantic affinity with image-level supervision for weakly supervised semantic segmentation." CVPR 2019.

2. **CAM**: Zhou, B., et al. (2016). "Learning deep features for discriminative localization." CVPR 2016.


## 📄 License

Kode ini dibuat untuk tujuan penelitian dan edukasi. Silakan gunakan dan modifikasi sesuai kebutuhan.

## 🙏 Acknowledgments

- Original IRNet implementation oleh Jiwoon Ahn
- ResNet50 pretrained weights dari PyTorch Model Zoo

