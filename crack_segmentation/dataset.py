"""
dataset.py - Dataset classes untuk crack detection dengan resolusi 4032x3024

Menyediakan:
1. CrackPatchDataset - Untuk training dengan patch extraction
2. CrackFullImageDataset - Untuk inference pada full image
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import config


class CrackPatchDataset(Dataset):
    """
    Dataset untuk training classification network (Stage 1).
    
    Ekstrak patch dari gambar besar (4032x3024) dengan sliding window.
    Setiap patch diklasifikasikan sebagai:
        - Positive (label=1): mengandung crack (ratio > MIN_CRACK_RATIO)
        - Negative (label=0): tidak ada crack
    """
    
    def __init__(self, img_dir, mask_dir, patch_size=512, stride=256, 
                 transform=None, max_neg_ratio=2.0, min_crack_ratio=0.03,
                 img_files=None):
        """
        Args:
            img_dir: Directory berisi crack images
            mask_dir: Directory berisi ground truth masks
            patch_size: Ukuran patch (default 512)
            stride: Stride untuk sliding window (default 256)
            transform: Transformasi augmentasi
            max_neg_ratio: Max ratio negative:positive patches
            min_crack_ratio: Min ratio crack pixels untuk dianggap positive
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.samples = []
        
        print(f"\n📦 Building CrackPatchDataset...")
        print(f"   Patch size: {patch_size}x{patch_size}")
        print(f"   Stride: {stride}")
        
        pos_patches = []
        neg_patches = []
        
        # Scan daftar gambar (bisa seluruh folder atau subset)
        if img_files is None:
            img_files = sorted([f for f in os.listdir(img_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            img_files = sorted(img_files)
        
        for fname in img_files:
            img_path = os.path.join(img_dir, fname)
            # Cari mask yang sesuai
            mask_name = fname.replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = os.path.join(mask_dir, mask_name)
            
            if not os.path.exists(mask_path):
                print(f"   ⚠ Warning: Mask not found for {fname}")
                continue
            
            # Load image dan mask
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                print(f"   ⚠ Warning: Failed to load {fname}")
                continue
            
            h, w = img.shape[:2]
            
            # Extract patches dengan sliding window
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    # Crop patch dari mask
                    patch_mask = mask[y:y+patch_size, x:x+patch_size]
                    
                    # Hitung ratio crack pixels (asumsi mask: 255=crack, 0=background)
                    crack_ratio = patch_mask.sum() / (patch_size * patch_size * 255.0)
                    
                    if crack_ratio > min_crack_ratio:
                        # Positive patch: ada crack
                        pos_patches.append((fname, x, y, 1))
                    elif crack_ratio == 0:
                        # Negative patch: pure background
                        neg_patches.append((fname, x, y, 0))
                    # Patches dengan sedikit crack (0 < ratio < min_crack_ratio) dibuang
        
        # Balance dataset: limit negative patches
        n_pos = len(pos_patches)
        n_neg = min(len(neg_patches), int(n_pos * max_neg_ratio))
        
        # Random sample negative patches
        if n_neg < len(neg_patches):
            import random
            neg_patches = random.sample(neg_patches, n_neg)
        
        self.samples = pos_patches + neg_patches
        
        print(f"   ✅ Dataset ready:")
        print(f"      Positive patches: {n_pos}")
        print(f"      Negative patches: {n_neg}")
        print(f"      Total: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Return (patch_tensor, label)"""
        fname, x, y, label = self.samples[idx]
        
        # Load full image
        img_path = os.path.join(self.img_dir, fname)
        img = cv2.imread(img_path)
        
        # Extract patch
        patch = img[y:y+self.patch_size, x:x+self.patch_size]
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        if self.transform:
            patch = self.transform(patch)
        
        return patch, label


class CrackFullImageDataset(Dataset):
    """
    Dataset untuk inference pada full image 4032x3024.
    
    Tidak ada patch extraction di sini - setiap item adalah full image.
    Patch extraction untuk inference dilakukan di inference loop.
    """
    
    def __init__(self, img_dir, mask_dir=None):
        """
        Args:
            img_dir: Directory berisi crack images
            mask_dir: Optional directory untuk ground truth (evaluasi)
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted([f for f in os.listdir(img_dir)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"\n📦 CrackFullImageDataset:")
        print(f"   Images: {len(self.img_files)}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            img_rgb: (H, W, 3) numpy array
            mask: (H, W) numpy array or None
            filename: str
        """
        fname = self.img_files[idx]
        img_path = os.path.join(self.img_dir, fname)
        
        # Load image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask if available
        mask = None
        if self.mask_dir is not None:
            mask_name = fname.replace('.jpg', '.png').replace('.jpeg', '.png')
            mask_path = os.path.join(self.mask_dir, mask_name)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        return img_rgb, mask, fname


# ============================================================
# DATA TRANSFORMS
# ============================================================

def get_train_transform():
    """Transform untuk training dengan augmentasi"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=config.AUG_H_FLIP_PROB),
        transforms.RandomVerticalFlip(p=config.AUG_V_FLIP_PROB),
        transforms.RandomRotation(config.AUG_ROTATION_DEGREES),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)
    ])


def get_inference_transform():
    """Transform untuk inference (no augmentation)"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)
    ])


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def denormalize_tensor(tensor):
    """Denormalize tensor untuk visualisasi"""
    mean = torch.tensor(config.IMG_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMG_STD).view(3, 1, 1)
    return tensor * std + mean


def get_image_splits(img_dir, n_train, n_val, n_test, seed=42):
    """Bagi dataset citra menjadi train/val/test berdasarkan jumlah citra.

    Split dilakukan di level gambar, sehingga semua patch yang berasal dari
    satu gambar hanya akan berada di satu subset (train/val/test).
    """

    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    total = len(img_files)
    target_total = n_train + n_val + n_test

    if total < target_total:
        raise ValueError(
            f"Not enough images in {img_dir}: found {total}, "
            f"but n_train+n_val+n_test={target_total}."
        )

    rng = np.random.RandomState(seed)
    indices = rng.permutation(total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:n_train + n_val + n_test]

    img_files = np.array(img_files)
    train_files = img_files[train_idx].tolist()
    val_files = img_files[val_idx].tolist()
    test_files = img_files[test_idx].tolist()

    print("\n🔀 Image-level split:")
    print(f"   Total images: {total}")
    print(f"   Train: {len(train_files)}")
    print(f"   Val:   {len(val_files)}")
    print(f"   Test:  {len(test_files)}")

    return train_files, val_files, test_files


def verify_dataset(img_dir, mask_dir):
    """
    Verifikasi dataset: check apakah semua images punya mask
    """
    print("\n🔍 Verifying dataset...")
    
    img_files = [f for f in os.listdir(img_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    matched = 0
    unmatched = []
    
    for fname in img_files:
        mask_name = fname.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            matched += 1
        else:
            unmatched.append(fname)
    
    print(f"   Total images: {len(img_files)}")
    print(f"   Matched with masks: {matched}")
    print(f"   Unmatched: {len(unmatched)}")
    
    if unmatched:
        print(f"   ⚠ Unmatched files: {unmatched[:5]}...")
    
    return matched, unmatched


if __name__ == "__main__":
    # Test dataset creation
    print("=" * 60)
    print("Testing Dataset Module")
    print("=" * 60)
    
    # Verify dataset
    verify_dataset(config.IMG_DIR, config.MASK_DIR)
    
    # Create patch dataset
    train_transform = get_train_transform()
    dataset = CrackPatchDataset(
        config.IMG_DIR, 
        config.MASK_DIR,
        patch_size=config.PATCH_SIZE,
        stride=config.PATCH_STRIDE,
        transform=train_transform,
        max_neg_ratio=config.MAX_NEG_RATIO,
        min_crack_ratio=config.MIN_CRACK_RATIO
    )
    
    print(f"\n✅ Dataset created successfully!")
    print(f"   Total patches: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample_img, sample_label = dataset[0]
        print(f"\n📊 Sample patch:")
        print(f"   Shape: {sample_img.shape}")
        print(f"   Label: {sample_label}")
        print(f"   Min/Max: {sample_img.min():.3f} / {sample_img.max():.3f}")
