# Installation Guide

Panduan lengkap instalasi dan setup untuk Crack Weakly Supervised Segmentation.

## 📋 Requirements

### System Requirements

- **OS**: Linux, macOS, atau Windows
- **Python**: 3.8 atau lebih baru
- **GPU**: NVIDIA GPU dengan CUDA support (recommended)
  - Minimum: 8GB VRAM
  - Recommended: 12GB+ VRAM
- **RAM**: Minimum 16GB
- **Storage**: ~10GB untuk kode, models, dan intermediate files

### Software Requirements

- Python 3.8+
- CUDA 11.0+ (untuk GPU training)
- pip atau conda

## 🚀 Installation Steps

### Step 1: Clone atau Download Repository

```bash
# Jika menggunakan git
git clone <repository-url>
cd crack_segmentation

# Atau download dan extract ZIP
unzip crack_segmentation.zip
cd crack_segmentation
```

### Step 2: Create Virtual Environment (Recommended)

#### Option A: Using venv

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

#### Option B: Using conda

```bash
# Create conda environment
conda create -n crack_seg python=3.9

# Activate
conda activate crack_seg
```

### Step 3: Install PyTorch

Install PyTorch sesuai dengan system Anda dari https://pytorch.org/get-started/locally/

#### For CUDA 11.8 (Recommended)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### For CUDA 12.1

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### For CPU Only

```bash
pip install torch torchvision
```

### Step 4: Install Other Dependencies

```bash
pip install -r requirements.txt
```

Atau install manual:

```bash
pip install numpy opencv-python scipy matplotlib tqdm Pillow
```

### Step 5: Install Optional Dependencies

#### DenseCRF (untuk post-processing)

```bash
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

#### Development Tools (optional)

```bash
pip install jupyter ipython tensorboard
```

### Step 6: Verify Installation

```bash
python quick_test.py
```

Script ini akan check:
- ✅ All dependencies installed correctly
- ✅ CUDA availability (jika ada GPU)
- ✅ Models can be instantiated
- ✅ Basic operations work

Expected output:

```
TEST SUMMARY
================================================================
Imports        : ✅ PASSED
Modules        : ✅ PASSED
Dataset        : ✅ PASSED
Models         : ✅ PASSED
CUDA           : ✅ PASSED
================================================================

🎉 All tests passed! You're ready to go!
```

## 📁 Setup Dataset

### Dataset Structure

Siapkan dataset dengan struktur berikut:

```
crack_segmentation/
└── data/
    ├── images/          # Gambar crack (JPG/PNG)
    │   ├── img001.jpg   # 4032x3024
    │   ├── img002.jpg
    │   └── ...
    └── masks/           # Ground truth masks (PNG)
        ├── img001.png   # Binary: 0=bg, 255=crack
        ├── img002.png
        └── ...
```

### Dataset Requirements

1. **Images**: Format JPG, JPEG, atau PNG
2. **Masks**: Format PNG, binary (0 dan 255)
3. **Naming**: Mask filename harus match dengan image
   - Image: `crack001.jpg` → Mask: `crack001.png`
   - Image: `photo.jpeg` → Mask: `photo.png`
4. **Resolution**: Seharusnya 4032x3024, tapi code bisa handle arbitrary sizes

### Example Dataset Preparation

```python
import os
import cv2
import numpy as np

# Buat folder
os.makedirs('data/images', exist_ok=True)
os.makedirs('data/masks', exist_ok=True)

# Contoh: Convert masks ke binary
mask_dir = 'data/masks'
for fname in os.listdir(mask_dir):
    if fname.endswith('.png'):
        mask = cv2.imread(os.path.join(mask_dir, fname), 0)
        # Threshold ke binary
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(mask_dir, fname), binary)
        print(f"Processed {fname}")
```

## ⚙️ Configuration

Edit `config.py` untuk menyesuaikan dengan setup Anda:

### Essential Settings

```python
# Dataset paths
IMG_DIR = "data/images"      # Path ke images
MASK_DIR = "data/masks"      # Path ke masks
OUTPUT_DIR = "outputs"       # Path untuk outputs

# Image resolution (sesuaikan dengan dataset Anda)
IMG_HEIGHT = 4032
IMG_WIDTH = 3024

# Training settings
CAM_EPOCHS = 25              # Epochs untuk CAM training
IRN_EPOCHS = 20              # Epochs untuk IRNet training
CAM_BATCH_SIZE = 8           # Reduce jika OOM
IRN_BATCH_SIZE = 4           # Reduce jika OOM
```

### Memory-Related Settings

Jika mengalami Out of Memory (OOM):

```python
# Reduce patch size
PATCH_SIZE = 384             # Default: 512
PATCH_STRIDE = 192           # Default: 256

# Reduce batch size
CAM_BATCH_SIZE = 4           # Default: 8
IRN_BATCH_SIZE = 2           # Default: 4

# Enable mixed precision
USE_AMP = True               # Default: True

# Reduce workers
NUM_WORKERS = 2              # Default: 4
```

## 🧪 Test Run

### Quick Test

```bash
# Test semua components
python quick_test.py

# Run example usage
python example_usage.py
```

### Mini Training Test

Test training dengan subset kecil data:

```python
# Edit config.py
CAM_EPOCHS = 2
IRN_EPOCHS = 2

# Run
python main.py --stage 1
```

Jika sukses, Anda siap untuk full training!

## 🔧 Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `PATCH_SIZE` dan `BATCH_SIZE` di `config.py`
2. Enable `USE_AMP = True`
3. Close other GPU applications
4. Use smaller model (reduce epochs)

### Issue 2: Import Error

**Symptoms**: `ModuleNotFoundError: No module named 'xxx'`

**Solutions**:
1. Verify virtual environment activated
2. Install missing package: `pip install xxx`
3. Reinstall requirements: `pip install -r requirements.txt`

### Issue 3: Dataset Not Found

**Symptoms**: `FileNotFoundError` atau `Dataset: 0 images`

**Solutions**:
1. Check path in `config.py` correct
2. Verify images exist: `ls data/images/`
3. Check file extensions (jpg, jpeg, png)
4. Run `python dataset.py` to verify

### Issue 4: Slow Training

**Symptoms**: Very slow iterations per second

**Solutions**:
1. Use GPU instead of CPU
2. Reduce `NUM_WORKERS` (try 0, 2, 4)
3. Increase `PATCH_STRIDE` (fewer patches)
4. Enable `PIN_MEMORY = True`

### Issue 5: Models Not Loading

**Symptoms**: `Checkpoint not found` errors

**Solutions**:
1. Train models first: `python main.py --stage 1`
2. Check `outputs/` directory exists
3. Verify model files exist:
   ```bash
   ls outputs/*.pth
   ```

## 📊 Verify Everything Works

### Complete Verification Checklist

- [ ] All dependencies installed (`pip list`)
- [ ] CUDA available (if using GPU)
- [ ] Dataset accessible (images + masks)
- [ ] Models can be instantiated
- [ ] Quick test passes
- [ ] Config.py settings correct
- [ ] Output directory writable

### Run Full Pipeline Test

```bash
# This will test everything
python main.py --stage all
```

Jika ini berhasil tanpa error, setup Anda sempurna! 🎉

## 🆘 Getting Help

Jika masih ada masalah:

1. Check error message carefully
2. Search in README.md for similar issues
3. Run `python quick_test.py` untuk diagnostic
4. Check CUDA compatibility
5. Verify dataset structure

## 📝 Next Steps

After successful installation:

1. ✅ Verify dataset dengan `python dataset.py`
2. ✅ Configure settings di `config.py`
3. ✅ Run quick test dengan `python quick_test.py`
4. ✅ Start training dengan `python main.py --stage all`
5. ✅ Monitor training di `outputs/`
6. ✅ Visualize results

Happy training! 🚀
