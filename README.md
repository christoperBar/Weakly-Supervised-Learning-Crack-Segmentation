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
          ↓
Stage 5: ResNet50-UNet Training
    ├─ Use pseudo labels as supervision
    └─ Train crack segmentation UNet (Stage 5)
          ↓
Stage 5v: UNet Visualization & Evaluation
    ├─ Plot training curves (loss, IoU, F1)
    └─ Visualize predictions vs ground truth on test set
```

## 📁 Project Structure (Current)

Struktur repo saat ini:

```
├── crack_segmentation/
│   ├── config.py                # Konfigurasi dataset & training
│   ├── dataset.py               # Dataset classes & data loading
│   ├── path_index.py            # PathIndex untuk IRNet
│   ├── resnet50.py              # ResNet50 backbone
│   ├── resnet50_cam.py          # CAM network
│   ├── resnet50_irn.py          # IRNet model (edge + displacement)
│   ├── train_stage1.py          # Stage 1: Train CAM
│   ├── train_stage2_3.py        # Stage 2+3: Mine relations + IRNet
│   ├── train_stage5.py          # Stage 5: Train ResNet50-UNet
│   ├── stage5_visualize.py      # Stage 5: Visualisasi hasil UNet
│   ├── inference.py             # Stage 4: Pseudo label synthesis
│   ├── main.py                  # Main pipeline script
│   ├── quick_test.py            # Quick sanity check environment
│   ├── example_usage.py         # Contoh penggunaan end-to-end
│   ├── requirements.txt         # Dependency Python (untuk pip)
│   │
│   ├── data/                    # (Opsional) salinan data lokal
│   │   ├── images/              # Crack images (4032x3024)
│   │   └── masks/               # Ground truth masks (binary)
│   │
│   └── outputs/                 # Output utama pipeline
│       ├── cam_net_best.pth     # Best CAM model
│       ├── cam_net_final.pth
│       ├── irnet_best.pth       # Best IRNet model
│       ├── irnet_final.pth
│       ├── pseudo_labels/       # Folder pseudo labels hasil
│       ├── stage5_unet/         # UNet dari pseudo labels CAM only
│       ├── stage5_unet_irn/     # UNet berbasis pseudo labels CAM+IRN 
│       └── visualizations/      # Semua visualisasi antar stage         
│
├── Dockerfile
├── run_docker.md
├── README.md
├── .gitignore
```

## 🧪 Test Run

### Quick Test

Semua perintah Python di bawah ini dijalankan dari folder `crack_segmentation/`.

```bash
cd crack_segmentation

# Test semua komponen & environment
python quick_test.py

# Run example usage (contoh-contoh umum)
python example_usage.py
```

## 🚀 Quick Start

### 1. Run Docker (Opsional)

Jalankan dari root repo (`crack_segmentation_docker/`):

```bash
docker run --gpus all -it \
    --shm-size=8g \
    -p 8000:8000 \
    -v "$(pwd -W)/crack_segmentation:/workspace/crack_segmentation" \
    -v "$(pwd -W)/data:/workspace/data" \
    crack-irn:py310
```

Detail tambahan lihat `run_docker.md`.

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

Edit `crack_segmentation/config.py` untuk menyesuaikan path dan parameters
(default-nya sudah diset sesuai struktur di atas):

```python
# Dataset paths (relatif terhadap root repo)
IMG_DIR = "data/images"      # Folder gambar
MASK_DIR = "data/masks"      # Folder ground truth mask
OUTPUT_DIR = "outputs"       # Akan dibuat di dalam crack_segmentation/

# Training parameters utama
CAM_EPOCHS = 50
IRN_EPOCHS = 50
PATCH_SIZE = 512
PATCH_STRIDE = 256
```

### 4. Run Pipeline

Semua perintah berikut dijalankan dari folder `crack_segmentation/`.

#### Option A: Run Full Pipeline

```bash
cd crack_segmentation
python main.py --stage all
```

#### Option B: Run Stage by Stage

```bash
cd crack_segmentation

# Stage 1: Train CAM network (classification + CAM)
python main.py --stage 1

# Stage 2+3: Mine inter-pixel relations + train IRNet
python main.py --stage 2

# Stage 4 (alias --stage 3 di kode): generate pseudo labels
python main.py --stage 3

# Jika hanya ingin inference (tanpa retrain)
python main.py --inference

# Stage 5: Train ResNet50-UNet dari pseudo labels
python main.py --stage 5

# Stage 5 Visualization: curve training + contoh segmentasi
python main.py --stage 5v
```
#### Option C: Run API

```bash
cd crack_segmentation

python api.py
```

## 🎨 Visualization

Setiap stage menghasilkan visualizations:

**Stage 1**: Training curves (loss, accuracy)

**Stage 2+3**: Loss decomposition (affinity, displacement)

**Stage 4** (pseudo label synthesis): 
- Original image
- CAM heatmap
- Boundary map
- Displacement field
- Pseudo label (CAM only and CAM+IRN)
- Overlay

**Stage 5**: 
- Training curves ResNet50-UNet (loss, IoU, F1)
- Prediksi vs ground truth di test set

Visualizations disimpan di `crack_segmentation/outputs/visualizations/` dan
subfolder terkait (misal: `stage5_unet/test_visualizations/`).

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

```

## 📚 References

1. **IRNet Paper**: Ahn, J., & Kwak, S. (2018). "Learning pixel-level semantic affinity with image-level supervision for weakly supervised semantic segmentation." CVPR 2019.

2. **CAM**: Zhou, B., et al. (2016). "Learning deep features for discriminative localization." CVPR 2016.


## 📄 License

Kode ini dibuat untuk tujuan penelitian dan edukasi. Silakan gunakan dan modifikasi sesuai kebutuhan.


