"""
ResNet50-UNet Crack Segmentation
=================================
Berdasarkan: Dong et al. (2020) - "Patch-based weakly supervised semantic 
segmentation network for crack detection"

PIPELINE YANG DIGUNAKAN:
    - Synthetic label berupa gambar FULL-SIZE (misal 4032x3024), bukan patch
    - Saat training: crop patch 224x224 secara on-the-fly dari pasangan (gambar, label)
    - Saat inference: sliding window pada gambar full-size -> neighborhood fusion

Dataset structure:
    data/
    ├── images/                 <- Gambar asli full-size (4032x3024)
    │   ├── img_001.jpg / .png
    │   └── ...
    └── labels/                 <- Synthetic label full-size (binary mask)
        ├── img_001.png         <- Nama file HARUS sama dengan images/
        └── ...

Usage:
    # Training
    python resnet50_unet_crack_segmentation.py train \
        --image_dir data/images \
        --label_dir data/labels \
        --output_dir checkpoints/

    # Inference pada gambar full-size
    python resnet50_unet_crack_segmentation.py infer \
        --image_path input/full_image.jpg \
        --checkpoint checkpoints/best_model.pth \
        --output_path output/result.png
"""

import os
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF

from PIL import Image
import random
import json
from tqdm import tqdm


# ============================================================
# 1. DATASET — Full-size image, crop patch on-the-fly
# ============================================================

class CrackFullImageDataset(Dataset):
    """
    Dataset yang menerima gambar full-size (misal 4032x3024) beserta
    synthetic label full-size, lalu crop patch 224x224 secara on-the-fly.

    Setiap __getitem__ menghasilkan SATU patch acak dari gambar yang dipilih.
    Dengan patches_per_img=20, satu gambar menghasilkan 20 sampel berbeda
    per epoch, memberikan variasi data yang besar.

    Args:
        image_dir      : folder gambar asli full-size
        label_dir      : folder synthetic label full-size (binary mask, 0/255)
        patch_size     : ukuran patch yang di-crop (default 224)
        patches_per_img: berapa patch yang di-sampling per gambar per epoch
        augment        : apakah augmentasi diaktifkan
        image_paths    : opsional, override daftar path gambar (untuk train/val split)
    """

    def __init__(self, image_dir, label_dir,
                 patch_size=224, patches_per_img=20, augment=True,
                 image_paths=None):
        self.image_dir       = Path(image_dir)
        self.label_dir       = Path(label_dir)
        self.patch_size      = patch_size
        self.patches_per_img = patches_per_img
        self.augment         = augment

        if image_paths is not None:
            self.image_paths = list(image_paths)
        else:
            exts = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]
            self.image_paths = []
            for ext in exts:
                self.image_paths += list(self.image_dir.glob(ext))
            self.image_paths = sorted(set(self.image_paths))

        assert len(self.image_paths) > 0, \
            f"Tidak ada gambar ditemukan di {image_dir}"

        # Verifikasi label tersedia
        missing = []
        for ip in self.image_paths:
            lp = self._get_label_path(ip)
            if not lp.exists():
                missing.append(f"  {ip.name} -> {lp}")
        if missing:
            raise FileNotFoundError(
                f"Label tidak ditemukan untuk {len(missing)} gambar:\n"
                + "\n".join(missing[:5])
                + ("\n  ..." if len(missing) > 5 else "")
            )

        # Normalisasi ImageNet untuk ResNet50
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        print(f"[Dataset] {len(self.image_paths)} gambar | "
              f"augment={augment} | "
              f"{len(self.image_paths) * patches_per_img} sampel/epoch")

    def _get_label_path(self, image_path):
        """Cari label di label_dir dengan nama stem yang sama."""
        # Selalu cari sebagai .png terlebih dahulu
        png_path = self.label_dir / (image_path.stem + ".png")
        if png_path.exists():
            return png_path
        # Fallback: nama file identik
        same_path = self.label_dir / image_path.name
        return same_path

    def __len__(self):
        return len(self.image_paths) * self.patches_per_img

    def __getitem__(self, idx):
        # Pilih gambar berdasarkan idx
        img_idx  = idx % len(self.image_paths)
        img_path = self.image_paths[img_idx]
        lbl_path = self._get_label_path(img_path)

        # Load
        image = Image.open(img_path).convert("RGB")
        label = Image.open(lbl_path).convert("L")

        # Pastikan ukuran label sama dengan gambar
        if image.size != label.size:
            label = label.resize(image.size, Image.NEAREST)

        # Random crop 224x224
        W, H = image.size
        P    = self.patch_size
        x    = random.randint(0, W - P)
        y    = random.randint(0, H - P)

        image_crop = image.crop((x, y, x + P, y + P))
        label_crop = label.crop((x, y, x + P, y + P))

        # Augmentasi
        if self.augment:
            image_crop, label_crop = self._augment(image_crop, label_crop)

        # Konversi ke tensor
        image_t = transforms.ToTensor()(image_crop)   # [3, P, P], float [0,1]
        image_t = self.normalize(image_t)

        label_np = np.array(label_crop)
        label_t  = torch.from_numpy(
            (label_np > 127).astype(np.int64)         # binarisasi
        )                                              # [P, P]

        return image_t, label_t

    def _augment(self, image, label):
        """Augmentasi simetris — diterapkan identik ke gambar dan label."""
        # Horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # Vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        # Rotasi 0/90/180/270
        angle = random.choice([0, 90, 180, 270])
        if angle:
            image = TF.rotate(image, angle)
            label = TF.rotate(label, angle)

        # Brightness & contrast (hanya image)
        image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        return image, label


# ============================================================
# 2. MODEL: ResNet50-UNet
# ============================================================

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample 2x -> Concat skip -> 2x ConvBnReLU"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=True)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet50UNet(nn.Module):
    """
    ResNet50-UNet sesuai paper Dong et al. (2020).

    Encoder (ResNet50 pretrained):
      e0  : Conv+BN+ReLU                  ->  64ch,   H/2
      pool: MaxPool                        ->  64ch,   H/4
      e1  : layer1                         -> 256ch,   H/4
      e2  : layer2                         -> 512ch,   H/8
      e3  : layer3                         ->1024ch,  H/16
      e4  : layer4                         ->2048ch,  H/32

    Bottleneck: 2048 -> 1024ch

    Decoder (dengan skip connection):
      D4: 1024 + skip(e3=1024) -> 512ch,  H/16
      D3:  512 + skip(e2=512)  -> 256ch,  H/8
      D2:  256 + skip(e1=256)  -> 128ch,  H/4
      D1:  128 + skip(e0=64)   ->  64ch,  H/2
      D0:   64 (tanpa skip)    ->  32ch,  H

    Head: Conv1x1 -> 2 kelas (background=0, crack=1)
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Encoder
        self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool      = backbone.maxpool
        self.encoder1  = backbone.layer1
        self.encoder2  = backbone.layer2
        self.encoder3  = backbone.layer3
        self.encoder4  = backbone.layer4

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBnRelu(2048, 1024),
            ConvBnRelu(1024, 1024),
        )

        # Decoder
        self.decoder4 = DecoderBlock(1024, 1024, 512)
        self.decoder3 = DecoderBlock(512,   512, 256)
        self.decoder2 = DecoderBlock(256,   256, 128)
        self.decoder1 = DecoderBlock(128,    64,  64)
        self.decoder0 = DecoderBlock(64,      0,  32)

        # Head
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e0 = self.encoder0(x)        # H/2,   64ch
        ep = self.pool(e0)           # H/4,   64ch
        e1 = self.encoder1(ep)       # H/4,  256ch
        e2 = self.encoder2(e1)       # H/8,  512ch
        e3 = self.encoder3(e2)       # H/16, 1024ch
        e4 = self.encoder4(e3)       # H/32, 2048ch

        b  = self.bottleneck(e4)     # H/32, 1024ch

        d4 = self.decoder4(b,  e3)  # H/16,  512ch
        d3 = self.decoder3(d4, e2)  # H/8,   256ch
        d2 = self.decoder2(d3, e1)  # H/4,   128ch
        d1 = self.decoder1(d2, e0)  # H/2,    64ch
        d0 = self.decoder0(d1)      # H,      32ch

        return self.head(d0)         # H, num_classes


# ============================================================
# 3. LOSS: Weighted Cross-Entropy (Eq. 5-6 paper)
# ============================================================

class WeightedCrossEntropyLoss(nn.Module):
    """
    alpha = N / N_positive  (Eq. 5)
    loss  = -alpha*y*log(p) - (1-alpha)*(1-y)*log(1-p)  (Eq. 6)

    Diimplementasikan via F.cross_entropy dengan weight=[1.0, alpha].
    """

    def forward(self, logits, labels):
        B, _, H, W = logits.shape
        N     = B * H * W
        N_pos = labels.sum().clamp(min=1).float()
        alpha = N / N_pos

        weight = torch.tensor([1.0, alpha.item()],
                               device=logits.device, dtype=torch.float32)
        return F.cross_entropy(logits, labels, weight=weight)


# ============================================================
# 4. METRICS (Eq. 7-12 paper)
# ============================================================

class SegmentationMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.TP = self.FP = self.FN = self.TN = 0

    def update(self, pred_np, label_np):
        pred  = pred_np.astype(bool)
        label = label_np.astype(bool)
        self.TP += int((pred  & label).sum())
        self.FP += int((pred  & ~label).sum())
        self.FN += int((~pred & label).sum())
        self.TN += int((~pred & ~label).sum())

    def compute(self):
        TP, FP, FN, TN = self.TP, self.FP, self.FN, self.TN
        eps = 1e-7
        return {
            "PA":   (TP + TN) / (TP + FP + FN + TN + eps),
            "MPA":  ((TP/(TP+FN+eps)) + (TN/(TN+FP+eps))) / 2,
            "MIoU": ((TP/(TP+FP+FN+eps)) + (TN/(TN+FN+FP+eps))) / 2,
            "P":    TP / (TP + FP + eps),
            "R":    TP / (TP + FN + eps),
            "F1":   2*TP / (2*TP + FP + FN + eps),
        }


# ============================================================
# 5. NEIGHBORHOOD FUSION — inference gambar full-size (Sec. 3.6)
# ============================================================

def neighborhood_fusion(model, image_np, device,
                         patch_size=224, step=80, batch_size=8):
    """
    Sliding window crop -> inferensi per patch -> gabung max-probability.

    Args:
        model      : ResNet50UNet (sudah eval mode)
        image_np   : numpy [H, W, 3], uint8
        device     : torch.device
        patch_size : ukuran patch (default 224, sesuai paper)
        step       : sliding step (default 80, sesuai paper)
        batch_size : jumlah patch per forward pass

    Returns:
        pred_mask  : numpy [H, W], uint8 (0=background, 1=crack)
        prob_map   : numpy [H, W], float32 (probabilitas kelas crack)
    """
    H, W = image_np.shape[:2]
    prob_acc = np.full((H, W), -np.inf, dtype=np.float32)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    to_tensor = transforms.ToTensor()

    # Buat koordinat sliding window (pastikan menyentuh tepi kanan & bawah)
    ys = list(range(0, H - patch_size, step)) + [max(0, H - patch_size)]
    xs = list(range(0, W - patch_size, step)) + [max(0, W - patch_size)]
    coords = [(y, x) for y in sorted(set(ys)) for x in sorted(set(xs))]

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(coords), batch_size),
                      desc="Neighborhood Fusion", leave=False):
            batch_coords  = coords[i:i + batch_size]
            batch_tensors = []
            for (y, x) in batch_coords:
                patch_np  = image_np[y:y+patch_size, x:x+patch_size]
                patch_pil = Image.fromarray(patch_np)
                batch_tensors.append(normalize(to_tensor(patch_pil)))

            batch    = torch.stack(batch_tensors).to(device)   # [B,3,P,P]
            logits   = model(batch)                              # [B,2,P,P]
            probs    = F.softmax(logits, dim=1)[:, 1]           # [B,P,P]
            probs_np = probs.cpu().numpy()

            for j, (y, x) in enumerate(batch_coords):
                # Max-probability fusion (Section 3.6)
                prob_acc[y:y+patch_size, x:x+patch_size] = np.maximum(
                    prob_acc[y:y+patch_size, x:x+patch_size],
                    probs_np[j]
                )

    prob_acc[prob_acc == -np.inf] = 0.0
    pred_mask = (prob_acc > 0.5).astype(np.uint8)
    return pred_mask, prob_acc


# ============================================================
# 6. TRAINING
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    metrics    = SegmentationMetrics()

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        for p, l in zip(preds, labels.cpu().numpy()):
            metrics.update(p, l)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader), metrics.compute()


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    metrics    = SegmentationMetrics()

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]  ", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        preds = logits.argmax(dim=1).cpu().numpy()
        for p, l in zip(preds, labels.cpu().numpy()):
            metrics.update(p, l)

    return total_loss / len(loader), metrics.compute()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ---------- Split train/val dari daftar gambar ----------
    exts = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"]
    all_paths = []
    for ext in exts:
        all_paths += list(Path(args.image_dir).glob(ext))
    all_paths = sorted(set(all_paths))

    random.seed(42)
    random.shuffle(all_paths)

    val_n       = max(1, int(0.2 * len(all_paths)))
    val_paths   = all_paths[:val_n]
    train_paths = all_paths[val_n:]
    print(f"[INFO] Train: {len(train_paths)} gambar | Val: {len(val_paths)} gambar")

    # ---------- Dataset & DataLoader ----------
    train_ds = CrackFullImageDataset(
        args.image_dir, args.label_dir,
        patch_size=args.patch_size,
        patches_per_img=args.patches_per_img,
        augment=True,
        image_paths=train_paths,
    )
    val_ds = CrackFullImageDataset(
        args.image_dir, args.label_dir,
        patch_size=args.patch_size,
        patches_per_img=args.patches_per_img,
        augment=False,
        image_paths=val_paths,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=args.num_workers,
                               pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=True)

    # ---------- Model, Loss, Optimizer ----------
    model     = ResNet50UNet(num_classes=2, pretrained=True).to(device)
    criterion = WeightedCrossEntropyLoss()

    # Adadelta sesuai paper Section 5.1
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    os.makedirs(args.output_dir, exist_ok=True)
    history   = {"train_loss": [], "val_loss": [], "val_miou": [], "val_f1": []}
    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        vl_loss, vl_m = validate(
            model, val_loader, criterion, device, epoch
        )
        scheduler.step(vl_m["MIoU"])

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_miou"].append(vl_m["MIoU"])
        history["val_f1"].append(vl_m["F1"])

        print(
            f"Epoch {epoch:3d}/{args.epochs}  |  "
            f"TrLoss={tr_loss:.4f}  VlLoss={vl_loss:.4f}  |  "
            f"MIoU={vl_m['MIoU']:.4f}  F1={vl_m['F1']:.4f}  "
            f"P={vl_m['P']:.4f}  R={vl_m['R']:.4f}"
        )

        if vl_m["MIoU"] > best_miou:
            best_miou = vl_m["MIoU"]
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": best_miou,
                "val_metrics": vl_m,
            }, ckpt_path)
            print(f"  --> Checkpoint disimpan: {ckpt_path}  (MIoU={best_miou:.4f})")

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[INFO] Training selesai. Best MIoU: {best_miou:.4f}")


# ============================================================
# 7. INFERENCE
# ============================================================

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = ResNet50UNet(num_classes=2, pretrained=False).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Checkpoint: epoch={ckpt['epoch']}, MIoU={ckpt['val_miou']:.4f}")

    image_pil = Image.open(args.image_path).convert("RGB")
    image_np  = np.array(image_pil)
    H, W      = image_np.shape[:2]
    print(f"[INFO] Input: {W}x{H} px")

    pred_mask, prob_map = neighborhood_fusion(
        model, image_np, device,
        patch_size=args.patch_size,
        step=args.step,
        batch_size=args.batch_size,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(out_dir, exist_ok=True)

    Image.fromarray((pred_mask * 255).astype(np.uint8)).save(args.output_path)
    print(f"[INFO] Mask tersimpan     : {args.output_path}")

    prob_path = args.output_path.replace(".png", "_prob.png")
    Image.fromarray((prob_map * 255).astype(np.uint8)).save(prob_path)
    print(f"[INFO] Prob map tersimpan : {prob_path}")
    print(f"[INFO] Crack coverage     : {pred_mask.mean()*100:.2f}%")


# ============================================================
# 8. ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ResNet50-UNet Crack Segmentation (Dong et al., 2020)\n"
                    "Input: gambar full-size + synthetic label full-size"
    )
    sub = parser.add_subparsers(dest="mode")

    # ---- train ----
    tr = sub.add_parser("train", help="Mode training")
    tr.add_argument("--image_dir",       required=True,
                    help="Folder gambar asli full-size")
    tr.add_argument("--label_dir",       required=True,
                    help="Folder synthetic label full-size (binary mask)")
    tr.add_argument("--output_dir",      default="checkpoints")
    tr.add_argument("--epochs",          type=int,   default=5,
                    help="Jumlah epoch (paper: 30)")
    tr.add_argument("--batch_size",      type=int,   default=8)
    tr.add_argument("--lr",              type=float, default=0.1,
                    help="Learning rate Adadelta (paper: 0.1)")
    tr.add_argument("--patch_size",      type=int,   default=224,
                    help="Ukuran patch (paper: 224x224)")
    tr.add_argument("--patches_per_img", type=int,   default=20,
                    help="Patch random yang di-sample per gambar per epoch")
    tr.add_argument("--num_workers",     type=int,   default=4)

    # ---- infer ----
    inf = sub.add_parser("infer", help="Mode inference gambar full-size")
    inf.add_argument("--image_path",  required=True,
                     help="Gambar input (bisa 4032x3024)")
    inf.add_argument("--checkpoint",  required=True,
                     help="Path checkpoint .pth")
    inf.add_argument("--output_path", default="output/result.png")
    inf.add_argument("--patch_size",  type=int, default=224)
    inf.add_argument("--step",        type=int, default=80,
                     help="Sliding step neighborhood fusion (paper: 80)")
    inf.add_argument("--batch_size",  type=int, default=8)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)
    else:
        print(__doc__)