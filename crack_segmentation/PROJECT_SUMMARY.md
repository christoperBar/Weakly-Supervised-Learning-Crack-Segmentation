# PROJECT SUMMARY – CRACK WEAKLY SUPERVISED SEGMENTATION (END‑TO‑END)

Dokumen ini merangkum **keseluruhan** proyek, dari konsep, data, arsitektur model, alur training Stage 1–5, sampai evaluasi dan deployment via Docker.

---

## 1. Latar Belakang & Tujuan

Deteksi retak (crack) pada struktur bangunan penting untuk pemeliharaan dan penilaian keselamatan. Namun, **label piksel (segmentation mask)** sulit dan mahal untuk dibuat secara manual pada citra resolusi tinggi (4032×3024).

Proyek ini mengusulkan pipeline **weakly supervised semantic segmentation** untuk crack detection berbasis framework **IRNet** (Ahn & Kwak, CVPR 2019), yang hanya membutuhkan **image‑level supervision** (informasi crack/no‑crack dari mask ground truth) untuk menghasilkan **pseudo segmentation mask** berkualitas tinggi, yang kemudian dipakai untuk melatih model segmentasi supervised (ResNet50‑UNet).

Tujuan utama:

1. Menghasilkan **peta segmentasi retak** pada citra resolusi tinggi.
2. Mengurangi ketergantungan pada **label piksel manual** dengan memanfaatkan **pseudo labels** dari IRNet.
3. Menyusun **pipeline end‑to‑end** yang dapat dijalankan dengan mudah (satu script) dan dapat di‑container‑kan dengan Docker.

---

## 2. Dataset & Pre‑processing

### 2.1 Struktur Dataset

Struktur dasar dataset yang digunakan:

```text

├── images/   # citra RGB 4032×3024
└── masks/    # ground truth mask biner (0=background, 255=crack)
```

Setelan default path dan parameter disimpan di [config.py](config.py):

- `IMG_DIR` → folder citra (mis. `data/images`)
- `MASK_DIR` → folder mask (mis. `data/masks`)
- `IMG_HEIGHT`, `IMG_WIDTH` → 4032×3024
- `OUTPUT_DIR` → direktori output global (mis. `outputs/`)

### 2.2 Patch‑based Training

Karena resolusi gambar besar, training dilakukan **berbasis patch**:

- `PATCH_SIZE` (default 512) dan `PATCH_STRIDE` (default 256) di [config.py](config.py).
- Fungsi ekstraksi patch dan dataset diimplementasikan di [dataset.py](dataset.py), terutama kelas `CrackPatchDataset` dan fungsi `get_image_splits`.
- Patches dibagi menjadi **train/val/test** berdasarkan jumlah citra, bukan per‑patch, untuk menjaga pemisahan data per‑gambar.
- Sampling patch mempertimbangkan **class imbalance**:
  - Parameter `MAX_NEG_RATIO` dan `MIN_CRACK_RATIO` mengontrol perbandingan patch tanpa crack vs patch dengan crack.

---

## 3. Arsitektur Model & Tahapan Utama

Secara garis besar, pipeline terdiri dari dua lapis besar:

1. **Weakly supervised tahap IRNet (Stage 1–4)** untuk menghasilkan **pseudo labels**.
2. **Fully supervised tahap ResNet50‑UNet (Stage 5)** yang dilatih menggunakan pseudo labels tersebut.

### 3.1 Komponen Inti

- [resnet50.py](resnet50.py) – ResNet50 backbone.
- [resnet50_cam.py](resnet50_cam.py) – model **CAM** (classification network) yang memetakan citra ke Class Activation Map.
- [resnet50_irn.py](resnet50_irn.py) – model **IRNet** dengan dua branch utama:
  - **Edge branch** → mempelajari peta boundary (B).
  - **Displacement branch** → mempelajari displacement field (D) antar piksel.
- [path_index.py](path_index.py) – membangun pasangan piksel (+/–) yang efisien untuk loss affinity IRNet.
- [tesunet.py](tesunet.py) – implementasi **ResNet50‑UNet** (Stage 5) untuk supervised crack segmentation.

### 3.2 Ringkasan Tahapan (Stage 1–5)

```text
Stage 1 : Train CAM (ResNet50 classifier)
Stage 2 : Mine inter‑pixel relations dari CAM
Stage 3 : Train IRNet
Stage 4 : Synthesize pseudo labels (CAM‑only & CAM+IRN)
Stage 5 : Train ResNet50‑UNet dengan pseudo labels
Stage 5v: Visualisasi & evaluasi model UNet
```

Seluruh tahapan diorkestrasi oleh [main.py](main.py).

---

## 4. Stage 1 – Classification Network (CAM)

**Tujuan:** melatih ResNet50‑based classifier yang tidak hanya membedakan crack/no‑crack, tapi juga menghasilkan **Class Activation Map (CAM)** untuk area retak.

Implementasi: [train_stage1.py](train_stage1.py)

### 4.1 Arsitektur & Loss

- Backbone: ResNet50.
- Classifier: 1×1 conv + global pooling untuk menghasilkan logits kelas.
- Loss: **Focal Loss** (kelas `FocalLoss` di [train_stage1.py](train_stage1.py)) untuk menangani **class imbalance**.

### 4.2 Proses Training

1. Patch training/validation di‑load melalui `CrackPatchDataset`.
2. Model `CamNet` dari [resnet50_cam.py](resnet50_cam.py) dilatih beberapa epoch (`CAM_EPOCHS`).
3. Setiap epoch:
   - Hitung loss & akurasi training serta validation.
4. Model terbaik disimpan sebagai **`cam_net_best.pth`**, dan model terakhir sebagai **`cam_net_final.pth`** di `OUTPUT_DIR`.
5. Riwayat training (loss & akurasi) divisualisasikan dan disimpan (mis. `outputs/stage1_training.png`).

Output utama Stage 1:

- Model CAM terlatih: `outputs/cam_net_best.pth`.
- Grafik training: loss & akurasi.

---

## 5. Stage 2–3 – IRNet: Inter‑pixel Relations & Edge/Displacement Learning

**Tujuan:** mempelajari **semantic affinity** antar piksel (apakah dua piksel termasuk region yang sama atau tidak) dan **displacement field** untuk pemisahan instance.

Implementasi utama: [train_stage2_3.py](train_stage2_3.py) dan [resnet50_irn.py](resnet50_irn.py).

### 5.1 Mining Inter‑Pixel Relations (Stage 2)

1. Dengan CAM terlatih, area dengan confidence tinggi dianggap sebagai **foreground seed**, dan area sangat rendah sebagai **background seed**.
2. [path_index.py](path_index.py) membentuk pasangan piksel (jalan pendek antar piksel) untuk:
   - **Positive pairs (P⁺)** – piksel yang seharusnya memiliki label sama.
   - **Negative pairs (P⁻)** – piksel yang seharusnya berada di sisi boundary berbeda.

### 5.2 Training IRNet (Stage 3)

IRNet terdiri dari:

- **Edge branch** → menghasilkan peta boundary B.
- **Displacement branch** → memprediksi vektor displacement D: setiap piksel diarahkan ke pusat regionnya.

Loss utama (disederhanakan):

- $L_{pos\_aff}$ – positive affinity loss.
- $L_{neg\_aff}$ – negative affinity (membatasi hubungan antara piksel beda instance).
- $L_{dp\_fg}$ & $L_{dp\_bg}$ – loss displacement untuk foreground dan background.

Total loss:  
$L_{total} = L_{pos\_aff} + L_{neg\_aff} + \gamma (L_{dp\_fg} + L_{dp\_bg})$,  
di mana $\gamma$ diatur di [config.py](config.py).

Output utama Stage 2–3:

- Model IRNet terlatih: `outputs/irnet_best.pth` dan `outputs/irnet_final.pth`.
- Grafik training: loss.

---

## 6. Stage 4 – Pseudo Label Synthesis (CAM‑only dan CAM+IRN)

**Tujuan:** mengubah sinyal CAM + IRNet menjadi **pseudo segmentation mask** resolusi penuh.

Implementasi: [inference.py](inference.py).

### 6.1 Sliding Window pada Full Image

Karena gambar berukuran 4032×3024, pemrosesan dilakukan dengan **sliding window patch‑based** (fungsi `process_full_image`):

1. Gambar dipecah menjadi patch ukuran `INFERENCE_PATCH_SIZE` dengan stride `INFERENCE_STRIDE` (keduanya diatur di [config.py](config.py)).
2. Setiap patch diproses oleh:
   - `extract_cam_for_patch` (CAMNet) → peta CAM lokal.
   - `extract_edge_displacement` (IRNet) → peta boundary B dan displacement D lokal.
3. Hasil tiap patch di‑blend kembali ke ukuran full image menggunakan **blending weight** halus untuk menghindari seam.

Output sementara:

- `cam_full` – CAM full‑image.
- `B_full` – boundary map full‑image.
- `D_full` – displacement field full‑image.

### 6.2 Synthesis CAM‑only

Untuk baseline CAM‑only:

1. `cam_full` dinormalisasi dan di‑enhance (gamma correction, smoothing).
2. Digunakan threshold otomatis (Otsu) dengan sedikit penyesuaian jika terlalu konservatif.
3. Dilakukan operasi morfologi ringan untuk membersihkan noise sambil menjaga struktur retak yang tipis.
4. Hasilnya adalah mask biner pseudo label **berbasis CAM saja**.

Output disimpan di:

- `outputs/pseudo_labels_cam_only/*.png`.

### 6.3 Hybrid CAM + IRNet (Boundary & Displacement)

Untuk pseudo label **CAM+IRN** (fungsi `synthesize_pseudo_label`):

1. Boundary map B digunakan untuk **refinement** CAM:
   - Area dekat seeds CAM yang kuat diperkaya kembali oleh sinyal IRNet (restorative fusion), bukan sekadar di‑suppress.
   - Hal ini membantu merekonstruksi retak yang tipis dan terputus.
2. **DenseCRF** digunakan (`apply_densecrf`) untuk edge‑aware smoothing menggunakan citra RGB asli (jika `USE_DENSECRF=True` dan library tersedia).
3. Thresholding + morfologi lanjutan = pseudo label binar final.


Output disimpan di:

- `outputs/pseudo_labels_cam_irn/*.png`.

### 6.4 Evaluasi Pseudo Label (IoU, Precision, Recall, F1)

Jika ground truth mask tersedia dan `COMPUTE_METRICS=True`, maka [inference.py](inference.py) menghitung:

- **IoU**, **Precision**, **Recall**, dan **F1** untuk setiap gambar, baik untuk **CAM‑only** maupun **CAM+IRN**.

File keluaran evaluasi:

- Per‑image:  
  `outputs/evaluation_metrics_per_image_cam_only.csv`  
  `outputs/evaluation_metrics_per_image_cam_irn.csv`
- Ringkasan (mean/std/min/max):  
  `outputs/evaluation_metrics_summary_cam_only.csv`  
  `outputs/evaluation_metrics_summary_cam_irn.csv`
- Plot distribusi dan rata‑rata metric:  
  `outputs/evaluation_metrics_overall_cam_only.png`  
  `outputs/evaluation_metrics_overall_cam_irn.png`  
  `outputs/evaluation_metrics_comparison_cam_vs_cam_irn.png`.

Secara umum, **CAM+IRN** diharapkan memberikan IoU dan F1 yang lebih tinggi dibanding CAM‑only, terutama pada retak yang tipis dan terputus.

---

## 7. Stage 5 – ResNet50‑UNet Supervised Training dari Pseudo Labels

**Tujuan:** melatih model segmentasi **ResNet50‑UNet** secara supervised menggunakan pseudo labels dari Stage 4 sebagai ground truth.

Implementasi: [train_stage5.py](train_stage5.py), [tesunet.py](tesunet.py), dan [stage5_visualize.py](stage5_visualize.py).

### 7.1 Sumber Data Stage 5

Beberapa konfigurasi penting di [config.py](config.py):

- `STAGE5_IMAGE_DIR` → folder citra untuk Stage 5 (umumnya sama dengan `IMG_DIR`).
- `STAGE5_LABEL_DIR` → folder pseudo labels (dipilih apakah pakai **CAM‑only** atau **CAM+IRN**).
- `STAGE5_OUTPUT_DIR` → folder output training UNet (mis. `outputs/stage5_unet/` atau `outputs/stage5_unet_irn/`).

Sebelum training, [train_stage5.py](train_stage5.py) memeriksa:

- Keberadaan folder citra dan pseudo label.
- Jumlah file citra dan label.

### 7.2 Training ResNet50‑UNet

1. Argumen training dibangun sebagai `Namespace` dan diteruskan ke fungsi `train_unet` di [tesunet.py](tesunet.py).
2. Hyperparameter utama:
   - `epochs` (`STAGE5_EPOCHS`).
   - `batch_size` (`STAGE5_BATCH_SIZE`).
   - `lr` (`STAGE5_LR`).
   - `patch_size` dan `patches_per_img` untuk training patch‑based di UNet.
3. Model terbaik disimpan sebagai **`best_model.pth`** dan riwayat training sebagai `history.json` di `STAGE5_OUTPUT_DIR`.

Output utama (contoh struktur):

```text
outputs/
└── stage5_unet_irn/
    ├── best_model.pth
    ├── history.json
    ├── test_metrics_summary.json
    ├── test_metrics_per_image.csv
    ├── test_predictions/
    └── test_visualizations/
```

File `test_metrics_summary.json` dan `test_metrics_per_image.csv` berisi metrik evaluasi (IoU, Precision, Recall, F1) untuk model UNet pada data uji.

### 7.3 Visualisasi Hasil Stage 5

Script [stage5_visualize.py](stage5_visualize.py) digunakan untuk:

- Menggambar kurva **loss**, **IoU**, dan **F1** selama training.
- Menyimpan contoh output segmentasi (prediksi vs ground truth) pada citra uji ke dalam:
  - `stage5_unet*/test_visualizations/`.

---

## 8. Orkestrasi Pipeline via main.py

File [main.py](main.py) menyatukan semua tahap di atas menjadi satu interface command‑line.

Contoh penggunaan utama:

```bash
# Menjalankan seluruh pipeline Stage 1–5
python main.py --stage all

# Menjalankan per stage
python main.py --stage 1   # Train CAM (Stage 1)
python main.py --stage 2   # Stage 2+3: IRNet
python main.py --inference # Stage 4: pseudo labels
python main.py --stage 5   # Stage 5: ResNet50‑UNet
python main.py --stage 5v  # Visualisasi Stage 5
```

`main.py` juga melakukan:

- **Verifikasi setup** dataset (folder gambar & mask) melalui `verify_setup()`.
- Menampilkan konfigurasi utama dari [config.py](config.py) melalui `print_config()`.
- Memastikan dependensi antar stage terpenuhi, misalnya:
  - `cam_net_best.pth` harus ada sebelum Stage 2–3.
  - `irnet_best.pth` harus ada sebelum Stage 4.
  - Folder pseudo labels harus ada sebelum Stage 5.

---

## 9. Struktur Output Utama

Direktori [outputs/](outputs) berisi semua hasil penting:

- `cam_net_best.pth`, `cam_net_final.pth` – model CAM.
- `irnet_best.pth`, `irnet_final.pth` – model IRNet.
- `pseudo_labels_cam_only/` – pseudo label berbasis CAM saja.
- `pseudo_labels_cam_irn/` – pseudo label hybrid CAM+IRN.
- `visualizations/` – visualisasi antar stage (CAM, boundary, overlay, dll.).
- `stage5_unet/` dan/atau `stage5_unet_irn/` – model UNet + log dan visualisasi.
- File CSV & PNG untuk **evaluation metrics** CAM‑only vs CAM+IRN.

---

## 10. Dockerization & Reproducibility

Untuk memastikan lingkungan yang konsisten, proyek ini dilengkapi dengan:

- [Dockerfile](../Dockerfile) – mendefinisikan image dengan Python, PyTorch, dan dependency lain.
- [run_docker.md](../run_docker.md) – instruksi menjalankan container, termasuk mounting folder `crack_segmentation/` dan `data/` ke dalam container, serta opsi `--gpus all` untuk akselerasi GPU.

Keuntungan Docker:

- Reproduksibilitas eksperimen.
- Memudahkan deploy ke mesin lain tanpa konflik environment.

---


## 11. Kesimpulan

Proyek ini merealisasikan **pipeline lengkap** crack segmentation berbasis **weakly supervised IRNet + supervised ResNet50‑UNet**, dengan fitur utama:

- Hanya membutuhkan **image‑level supervision** untuk membangkitkan **pseudo segmentation mask** berkualitas tinggi.
- Mampu menangani **citra resolusi tinggi (4032×3024)** dengan pendekatan patch‑based yang efisien.
- Menyediakan tahapan **Stage 1–5** yang jelas, dari training classifier, IRNet, pembuatan pseudo label, hingga training model segmentasi akhir dan evaluasinya.
- Didukung dengan **visualisasi komprehensif** dan **evaluasi kuantitatif** (IoU, Precision, Recall, F1) untuk membandingkan CAM‑only vs CAM+IRN dan performa model UNet.
- Siap dijalankan di lingkungan ter‑container (Docker) untuk riset lanjutan maupun integrasi ke pipeline produksi.

Dengan demikian, proyek ini dapat menjadi **fondasi** untuk penelitian dan pengembangan lebih lanjut di bidang deteksi retak dan weakly supervised semantic segmentation.

