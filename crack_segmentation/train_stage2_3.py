"""
train_stage2_3.py - Stage 2+3: Mine Inter-Pixel Relations & Train IRNet

Stage 2: Extract CAMs dan mine inter-pixel relations (P+, P-)
Stage 3: Train IRNet (edge + displacement branches) dengan supervision dari relations

Output: IRNet model yang menghasilkan boundary map dan displacement field
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

import config
from resnet50_cam import CAM as CamExtractor
from resnet50_irn import AffinityDisplacementLoss
from path_index import PathIndex
from dataset import get_inference_transform, get_image_splits


def spatial_mask_to_pairs(mask, path_index, debug=False):
    """
    Convert spatial mask (H, W) to pair-wise mask matching PathIndex structure.
    
    Args:
        mask: (B, H, W) binary mask tensor
        path_index: PathIndex object defining pixel pairs
    
    Returns:
        pair_mask: (B, num_pairs) binary mask where 1 = both pixels in pair are in mask
    """
    B, H, W = mask.shape
    device = mask.device
    mask_flat = mask.view(B, -1)  # (B, H*W)
    
    if debug:
        print(f"[spatial_mask_to_pairs] Input mask shape: {mask.shape}")
        print(f"[spatial_mask_to_pairs] PathIndex size: {path_index.feat_h}x{path_index.feat_w}")
        print(f"[spatial_mask_to_pairs] PathIndex directions: {len(path_index.search_dst)}")
    
    pair_masks_list = []
    
    for dir_idx, (dy, dx) in enumerate(path_index.search_dst):
        # Valid source region (so dst stays in-bounds)
        y_start = max(0, -dy)
        y_end = min(H, H - dy)
        x_start = max(0, -dx)
        x_end = min(W, W - dx)
        
        # Create indices for all valid source pixels at once
        src_indices = []
        dst_indices = []
        
        for sy in range(y_start, y_end):
            for sx in range(x_start, x_end):
                src_idx = sy * W + sx
                dst_idx = (sy + dy) * W + (sx + dx)
                src_indices.append(src_idx)
                dst_indices.append(dst_idx)
        
        # Batch gather
        src_indices_t = torch.tensor(src_indices, device=device, dtype=torch.long)
        dst_indices_t = torch.tensor(dst_indices, device=device, dtype=torch.long)
        
        src_vals = torch.gather(mask_flat, 1, src_indices_t.unsqueeze(0).expand(B, -1))  # (B, N)
        dst_vals = torch.gather(mask_flat, 1, dst_indices_t.unsqueeze(0).expand(B, -1))  # (B, N)
        
        # Pair is valid only if BOTH pixels are in the mask
        pair_vals = src_vals * dst_vals  # (B, N)
        pair_masks_list.append(pair_vals)
        
        if debug and dir_idx == 0:
            print(f"[spatial_mask_to_pairs] Direction 0: ({dy}, {dx}), pairs: {len(src_indices)}")
    
    # Concatenate all directions: (B, num_pairs)
    pair_mask = torch.cat(pair_masks_list, dim=1)
    
    if debug:
        print(f"[spatial_mask_to_pairs] Total pair_mask shape: {pair_mask.shape}")
    
    return pair_mask


def spatial_masks_to_cross_pairs(src_mask, dst_mask, path_index):
    """
    Build cross-class pair mask for affinity loss.

    A pair is valid if source pixel is in src_mask AND destination pixel is in dst_mask.
    This is used for negative affinity supervision (fg <-> bg pairs).
    """
    B, H, W = src_mask.shape
    device = src_mask.device

    src_flat = src_mask.view(B, -1)
    dst_flat = dst_mask.view(B, -1)

    pair_masks_list = []

    for dy, dx in path_index.search_dst:
        y_start = max(0, -dy)
        y_end = min(H, H - dy)
        x_start = max(0, -dx)
        x_end = min(W, W - dx)

        src_indices = []
        dst_indices = []

        for sy in range(y_start, y_end):
            for sx in range(x_start, x_end):
                src_idx = sy * W + sx
                dst_idx = (sy + dy) * W + (sx + dx)
                src_indices.append(src_idx)
                dst_indices.append(dst_idx)

        src_indices_t = torch.tensor(src_indices, device=device, dtype=torch.long)
        dst_indices_t = torch.tensor(dst_indices, device=device, dtype=torch.long)

        src_vals = torch.gather(src_flat, 1, src_indices_t.unsqueeze(0).expand(B, -1))
        dst_vals = torch.gather(dst_flat, 1, dst_indices_t.unsqueeze(0).expand(B, -1))

        pair_vals = src_vals * dst_vals
        pair_masks_list.append(pair_vals)

    return torch.cat(pair_masks_list, dim=1)


def spatial_masks_to_directional_pairs(src_mask, dst_mask, path_index):
    """
    Build directional pair masks aligned with displacement tensor layout.

    Returns:
        pair_mask: (B, D, N) where D=len(search_dst), N=cropped_H*cropped_W
    """
    B, H, W = src_mask.shape
    radius_floor = path_index.radius_floor
    cropped_h = H - radius_floor
    cropped_w = W - 2 * radius_floor

    src_crop = src_mask[:, :cropped_h, radius_floor:radius_floor + cropped_w]

    directional_pairs = []
    for dy, dx in path_index.search_dst:
        dst_crop = dst_mask[:, dy:dy + cropped_h, radius_floor + dx:radius_floor + dx + cropped_w]
        directional_pairs.append(src_crop * dst_crop)

    pair_mask = torch.stack(directional_pairs, dim=1).reshape(B, len(path_index.search_dst), -1)
    return pair_mask


class IRNetDataset(Dataset):
    """
    Dataset untuk IRNet training.
    
    Untuk setiap image, kita perlu:
    1. Patch RGB (256x256)
    2. CAM untuk patch
    3. Inter-pixel relations (P+_fg, P+_bg, P-) dari CAM
    """
    
    def __init__(self, img_dir, cam_net, device, patch_size=256, img_files=None):
        self.img_dir = img_dir
        self.cam_net = cam_net
        self.device = device
        self.patch_size = patch_size
        self.transform = get_inference_transform()
        
        # Collect image files (bisa seluruh folder atau subset train)
        if img_files is None:
            self.img_files = sorted([f for f in os.listdir(img_dir)
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            self.img_files = sorted(img_files)
        
        # Pre-extract patches with their CAMs
        self.samples = []
        self._extract_patches()
    
    def _extract_patches(self):
        """Extract semua patches dan CAMs"""
        print(f"\n📦 Extracting patches and CAMs...")
        
        for fname in tqdm(self.img_files, desc="Processing images"):
            img_path = os.path.join(self.img_dir, fname)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h, w = img_rgb.shape[:2]
            ps = self.patch_size
            
            # Sliding window
            for y in range(0, h - ps + 1, ps // 2):  # 50% overlap
                for x in range(0, w - ps + 1, ps // 2):
                    patch_rgb = img_rgb[y:y+ps, x:x+ps]
                    
                    # Extract CAM for this patch
                    cam = self._extract_cam(patch_rgb)
                    
                    # Only keep patches with meaningful crack signal
                    crack_ratio = (cam > 0.4).sum() / cam.size
                    if crack_ratio > 0.05 and crack_ratio < 0.95:  # Has crack but not too much
                        self.samples.append({
                            'fname': fname,
                            'x': x,
                            'y': y,
                            'patch_rgb': patch_rgb,
                            'cam': cam
                        })
        
        print(f"   ✅ Total patches: {len(self.samples)}")
    
    def _extract_cam(self, patch_rgb):
        """Extract CAM untuk single patch"""
        self.cam_net.eval()
        
        # Prepare input: [original, h-flipped]
        t_orig = self.transform(patch_rgb)
        t_flip = t_orig.flip(-1)
        batch = torch.stack([t_orig, t_flip], dim=0).to(self.device)
        
        with torch.no_grad():
            x = self.cam_net.stage1(batch)
            x = self.cam_net.stage2(x)
            x = self.cam_net.stage3(x)
            x = self.cam_net.stage4(x)
            x = F.conv2d(x, self.cam_net.classifier.weight)
            x = F.relu(x)
            
            # TTA: average original and flipped
            cam_2d = x[0] + x[1].flip(-1)  # (NUM_CLASSES, h, w)
        
        # Take crack class (index 1)
        cam = cam_2d[1].cpu().numpy()
        
        # Normalize to [0, 1]
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            patch_tensor: (3, 256, 256) normalized
            cam_tensor: (64, 64) CAM tensor
            pos_fg_mask: (64, 64) positive foreground mask
            pos_bg_mask: (64, 64) positive background mask
            neg_mask: (64, 64) negative mask
        """
        sample = self.samples[idx]
        patch_rgb = sample['patch_rgb']
        cam = sample['cam']
        
        # Transform patch to tensor
        patch_tensor = self.transform(patch_rgb)
        
        # Mine inter-pixel relations from CAM (HIGHER thresholds for thin cracks)
        relations = mine_relations(cam, fg_thresh=0.6, bg_thresh=0.25)
        
        # Convert to tensors
        cam_tensor = torch.from_numpy(cam).float()
        pos_fg_tensor = torch.from_numpy(relations['pos_fg']).float()
        pos_bg_tensor = torch.from_numpy(relations['pos_bg']).float()
        neg_tensor = torch.from_numpy(relations['neg']).float()
        
        return patch_tensor, cam_tensor, pos_fg_tensor, pos_bg_tensor, neg_tensor


def mine_relations(cam, fg_thresh=0.6, bg_thresh=0.2):
    """
    Mine inter-pixel relations dari CAM - IMPROVED VERSION.
    
    Args:
        cam: (H, W) CAM array
        fg_thresh: Threshold untuk foreground (high confidence)
        bg_thresh: Threshold untuk background (low confidence)
    
    Returns:
        dict dengan keys:
            'pos_fg': (H, W) binary mask untuk P+_fg pairs (confident crack)
            'pos_bg': (H, W) binary mask untuk P+_bg pairs (confident background)
            'neg': (H, W) binary mask untuk P- pairs (boundaries)
    """
    from scipy.ndimage import binary_erosion, binary_dilation
    
    h, w = cam.shape
    
    # More aggressive thresholding for better separation
    fg_mask = cam > fg_thresh  # Very confident foreground
    bg_mask = cam < bg_thresh  # Very confident background
    
    # Erode masks to get only high-confidence core regions
    fg_core = binary_erosion(fg_mask, iterations=2)
    bg_core = binary_erosion(bg_mask, iterations=2)
    
    # Dilate to find boundaries
    fg_dilated = binary_dilation(fg_core, iterations=3)
    bg_dilated = binary_dilation(bg_core, iterations=3)
    
    # Boundary = where dilated fg and bg overlap
    boundary = fg_dilated & bg_dilated
    
    # Also use gradient
    grad_y = np.abs(cam[1:, :] - cam[:-1, :])
    grad_x = np.abs(cam[:, 1:] - cam[:, :-1])
    grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
    grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)
    high_grad = grad_mag > 0.2
    
    # Combine for better boundary detection
    boundary_final = boundary | high_grad
    
    # Ensure we have enough samples
    pos_fg = fg_core.astype(np.float32)
    pos_bg = bg_core.astype(np.float32)
    neg = boundary_final.astype(np.float32)
    
    # If too sparse, fall back to softer thresholds
    if pos_fg.sum() < 100:
        pos_fg = (cam > 0.5).astype(np.float32)
    if pos_bg.sum() < 100:
        pos_bg = (cam < 0.3).astype(np.float32)
    if neg.sum() < 50:
        neg = ((cam >= 0.3) & (cam <= 0.5)).astype(np.float32)
    
    return {
        'pos_fg': pos_fg,
        'pos_bg': pos_bg,
        'neg': neg
    }


def _grad_group_l2_norm(params):
    """Compute L2 norm of gradients for a parameter group."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            total += g.pow(2).sum().item()
    return float(total ** 0.5)


def train_irnet_epoch(model, loader, optimizer, device, epoch, gamma=5.0, edge_params=None, dp_params=None):
    """Train IRNet satu epoch dengan proper pair-wise masking"""
    model.train()
    
    total_loss = 0
    total_pos_aff = 0
    total_neg_aff = 0
    total_dp_fg = 0
    total_dp_bg = 0
    total_pos_fg_density = 0
    total_pos_bg_density = 0
    total_neg_density = 0
    total_edge_grad_norm = 0
    total_dp_grad_norm = 0
    valid_batches = 0
    
    # Get path_index from model for mask conversion
    path_index = model.path_index
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (imgs, cams, pos_fg_masks, pos_bg_masks, neg_masks) in enumerate(pbar):
        imgs = imgs.to(device)
        cams = cams.to(device)
        pos_fg_masks = pos_fg_masks.to(device)
        pos_bg_masks = pos_bg_masks.to(device)
        neg_masks = neg_masks.to(device)
        
        # Forward pass dengan loss computation
        optimizer.zero_grad()
        
        # IRNet forward returns raw loss tensors
        # pos_aff_loss_raw, neg_aff_loss_raw: (B, num_pairs)
        # dp_fg_loss_raw, dp_bg_loss_raw: (B, 2, num_directions, N) where N = cropped_H * cropped_W
        pos_aff_loss_raw, neg_aff_loss_raw, dp_fg_loss_raw, dp_bg_loss_raw = model(imgs, True)
        
        # CRITICAL FIX: Masks are ALREADY at feature map resolution (CAM resolution)
        # Don't downsample them further! They should match PathIndex size.
        # The relation masks come from CAM which has the same backbone as IRNet
        
        # Debug info
        if batch_idx == 0:
            print(f"\n[Debug] Input image shape: {imgs.shape}")
            print(f"[Debug] Relation mask shape: {pos_fg_masks.shape}")
            print(f"[Debug] PathIndex size: {path_index.feat_h}x{path_index.feat_w}")
            print(f"[Debug] pos_aff_loss_raw shape: {pos_aff_loss_raw.shape}")
        
        # Ensure masks match PathIndex size
        B, H, W = pos_fg_masks.shape
        expected_h, expected_w = path_index.feat_h, path_index.feat_w
        
        if H != expected_h or W != expected_w:
            # Resize masks to match PathIndex size
            pos_fg_masks_resized = F.interpolate(pos_fg_masks.unsqueeze(1).float(), 
                                                 size=(expected_h, expected_w), 
                                                 mode='nearest').squeeze(1)
            pos_bg_masks_resized = F.interpolate(pos_bg_masks.unsqueeze(1).float(), 
                                                 size=(expected_h, expected_w), 
                                                 mode='nearest').squeeze(1)
            neg_masks_resized = F.interpolate(neg_masks.unsqueeze(1).float(), 
                                             size=(expected_h, expected_w), 
                                             mode='nearest').squeeze(1)
        else:
            pos_fg_masks_resized = pos_fg_masks
            pos_bg_masks_resized = pos_bg_masks
            neg_masks_resized = neg_masks
        
        # Convert to pair-wise masks matching PathIndex structure
        pos_fg_pair = spatial_mask_to_pairs(pos_fg_masks_resized, path_index, debug=(batch_idx==0))
        pos_bg_pair = spatial_mask_to_pairs(pos_bg_masks_resized, path_index)

        # Negative affinity should come from cross-class pairs (fg<->bg).
        neg_pair_fg_bg = spatial_masks_to_cross_pairs(pos_fg_masks_resized, pos_bg_masks_resized, path_index)
        neg_pair_bg_fg = spatial_masks_to_cross_pairs(pos_bg_masks_resized, pos_fg_masks_resized, path_index)
        neg_pair = torch.clamp(neg_pair_fg_bg + neg_pair_bg_fg, max=1.0)
        
        # Debug info  
        if batch_idx == 0:
            print(f"\n[Debug] pos_fg_pair shape: {pos_fg_pair.shape}")
            print(f"[Debug] Pair masks - FG: {pos_fg_pair.sum().item()}/{pos_fg_pair.numel()}, "
                  f"BG: {pos_bg_pair.sum().item()}/{pos_bg_pair.numel()}, "
                  f"NEG: {neg_pair.sum().item()}/{neg_pair.numel()}")
        
        # Apply pair-wise masks to affinity losses
        # Squeeze the extra dimension from loss if present
        if pos_aff_loss_raw.dim() == 3:
            pos_aff_loss_raw = pos_aff_loss_raw.squeeze(-1)  # (B, num_pairs)
        if neg_aff_loss_raw.dim() == 3:
            neg_aff_loss_raw = neg_aff_loss_raw.squeeze(-1)  # (B, num_pairs)
        
        # Positive affinity uses both confident foreground and confident background pairs.
        pos_aff_loss_masked_fg = pos_aff_loss_raw * pos_fg_pair
        pos_aff_loss_masked_bg = pos_aff_loss_raw * pos_bg_pair
        neg_aff_loss_masked = neg_aff_loss_raw * neg_pair     # (B, num_pairs)
        
        # Average only over valid (non-zero) pairs
        pos_fg_count = pos_fg_pair.sum() + 1e-5
        pos_bg_count = pos_bg_pair.sum() + 1e-5
        pos_count = pos_fg_count + pos_bg_count
        neg_count = neg_pair.sum() + 1e-5
        
        pos_aff_loss = (pos_aff_loss_masked_fg.sum() + pos_aff_loss_masked_bg.sum()) / pos_count
        neg_aff_loss = neg_aff_loss_masked.sum() / neg_count
        
        # For displacement losses, mask directly at directional pair level.
        # dp_*_loss_raw shape: (B, 2, D, N)
        dp_fg_pair_mask = spatial_masks_to_directional_pairs(pos_fg_masks_resized, pos_fg_masks_resized, path_index)
        dp_bg_pair_mask = spatial_masks_to_directional_pairs(pos_bg_masks_resized, pos_bg_masks_resized, path_index)

        dp_fg_mask_exp = dp_fg_pair_mask.unsqueeze(1)  # (B, 1, D, N)
        dp_bg_mask_exp = dp_bg_pair_mask.unsqueeze(1)  # (B, 1, D, N)

        dp_fg_count = dp_fg_mask_exp.sum() * dp_fg_loss_raw.size(1) + 1e-5
        dp_bg_count = dp_bg_mask_exp.sum() * dp_bg_loss_raw.size(1) + 1e-5

        dp_fg_loss = (dp_fg_loss_raw * dp_fg_mask_exp).sum() / dp_fg_count
        dp_bg_loss = (dp_bg_loss_raw * dp_bg_mask_exp).sum() / dp_bg_count

        # Keep density logging for visibility
        pos_fg_density = pos_fg_pair.float().mean(dim=1, keepdim=True)  # (B, 1)
        pos_bg_density = pos_bg_pair.float().mean(dim=1, keepdim=True)  # (B, 1)
        neg_density = neg_pair.float().mean(dim=1, keepdim=True)        # (B, 1)
        
        # Small regularization on displacement magnitude
        dp_reg = 0.001 * (dp_fg_loss_raw.pow(2).mean() + dp_bg_loss_raw.pow(2).mean())
        
        # Compute total loss (Eq 9 in IRNet paper)
        # L = L_pos + L_neg + γ(L_dp_fg + L_dp_bg) + regularization
        loss = (pos_aff_loss + 
                neg_aff_loss + 
                gamma * (dp_fg_loss + dp_bg_loss) +
                dp_reg)
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  Warning: Invalid loss detected at batch {batch_idx}, skipping...")
            continue
        
        # Backward with gradient clipping
        loss.backward()
        edge_grad_norm = _grad_group_l2_norm(edge_params) if edge_params is not None else 0.0
        dp_grad_norm = _grad_group_l2_norm(dp_params) if dp_params is not None else 0.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        total_pos_aff += pos_aff_loss.item()
        total_neg_aff += neg_aff_loss.item()
        total_dp_fg += dp_fg_loss.item()
        total_dp_bg += dp_bg_loss.item()
        total_pos_fg_density += pos_fg_density.mean().item()
        total_pos_bg_density += pos_bg_density.mean().item()
        total_neg_density += neg_density.mean().item()
        total_edge_grad_norm += edge_grad_norm
        total_dp_grad_norm += dp_grad_norm
        valid_batches += 1
        
        # Update progress
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pos': f'{pos_aff_loss.item():.4f}',
            'neg': f'{neg_aff_loss.item():.4f}',
            'dp_fg': f'{dp_fg_loss.item():.3f}',
            'dp_bg': f'{dp_bg_loss.item():.3f}',
            'g_e': f'{edge_grad_norm:.3f}',
            'g_d': f'{dp_grad_norm:.3f}'
        })
    
    n = max(valid_batches, 1)
    
    return {
        'total_loss': total_loss / n,
        'pos_aff': total_pos_aff / n,
        'neg_aff': total_neg_aff / n,
        'dp_fg': total_dp_fg / n,
        'dp_bg': total_dp_bg / n,
        'pos_fg_density': total_pos_fg_density / n,
        'pos_bg_density': total_pos_bg_density / n,
        'neg_density': total_neg_density / n,
        'edge_grad_norm': total_edge_grad_norm / n,
        'dp_grad_norm': total_dp_grad_norm / n,
        'valid_batches': valid_batches
    }


def plot_irnet_history(history, save_path='outputs/stage2_3_training.png'):
    """Plot IRNet training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    axes[0, 0].plot(history['total_loss'], marker='o')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Affinity losses
    axes[0, 1].plot(history['pos_aff'], label='Positive', marker='o')
    axes[0, 1].plot(history['neg_aff'], label='Negative', marker='s')
    axes[0, 1].set_title('Affinity Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Displacement losses
    axes[1, 0].plot(history['dp_fg'], label='Foreground', marker='o')
    axes[1, 0].plot(history['dp_bg'], label='Background', marker='s')
    axes[1, 0].set_title('Displacement Losses')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # All losses combined
    axes[1, 1].plot(history['total_loss'], label='Total', marker='o')
    axes[1, 1].plot(history['pos_aff'], label='Pos Aff', alpha=0.7)
    axes[1, 1].plot(history['neg_aff'], label='Neg Aff', alpha=0.7)
    axes[1, 1].set_title('All Losses')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Training plot saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("STAGE 2+3: Mine Relations & Train IRNet")
    print("=" * 60)
    
    device = config.DEVICE
    print(f"\n🖥️  Device: {device}")
    
    # Load trained CAM network
    print(f"\n📥 Loading CAM network...")
    cam_net = CamExtractor().to(device)
    cam_path = os.path.join(config.OUTPUT_DIR, 'cam_net_best.pth')
    
    if not os.path.exists(cam_path):
        print(f"❌ Error: CAM network not found at {cam_path}")
        print(f"   Please run train_stage1.py first!")
        return
    
    checkpoint = torch.load(cam_path, map_location=device, weights_only=False)
    cam_net.load_state_dict(checkpoint['model_state_dict'])
    cam_net.eval()
    print(f"   ✅ Loaded from {cam_path}")
    
    # Create IRNet dataset
    print(f"\n📦 Creating IRNet dataset...")
    irn_patch_size = 256  # IRNet uses smaller patches than CAM stage

    # Gunakan subset citra train untuk IRNet (jaga val/test tetap terpisah)
    train_imgs, val_imgs, test_imgs = get_image_splits(
        config.IMG_DIR,
        n_train=config.N_TRAIN_IMAGES,
        n_val=config.N_VAL_IMAGES,
        n_test=config.N_TEST_IMAGES,
        seed=config.RANDOM_SEED
    )

    dataset = IRNetDataset(
        img_dir=config.IMG_DIR,
        cam_net=cam_net,
        device=device,
        patch_size=irn_patch_size,
        img_files=train_imgs
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.IRN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"   Patch size: {irn_patch_size}x{irn_patch_size}")
    print(f"   Batch size: {config.IRN_BATCH_SIZE}")
    print(f"   Batches per epoch: {len(loader)}")
    
    # Create IRNet model
    print(f"\n🏗️  Building IRNet...")
    # PathIndex must match the ACTUAL feature map size from IRNetDataset
    irn_feat_h = irn_patch_size // 4  # 64 for 256x256 patches
    irn_feat_w = irn_patch_size // 4  # 64 for 256x256 patches
    path_idx = PathIndex(
        feat_h=irn_feat_h,
        feat_w=irn_feat_w,
        radius=config.RADIUS
    )
    
    model = AffinityDisplacementLoss(path_idx).to(device)
    
    # Optimizer (only for branch parameters, backbone frozen)
    edge_params, dp_params = model.trainable_parameters()
    edge_params = list(edge_params)
    dp_params = list(dp_params)
    
    # Separate learning rates for edge and displacement branches
    param_groups = [
        {'params': edge_params, 'lr': config.IRN_LR, 'name': 'edge'},
        {'params': dp_params, 'lr': config.IRN_LR * 0.5, 'name': 'displacement'}  # Lower LR for displacement
    ]
    
    optimizer = torch.optim.SGD(
        param_groups,
        momentum=0.9,
        weight_decay=config.IRN_WEIGHT_DECAY
    )
    
    # Warmup + Cosine annealing scheduler
    def lr_lambda(epoch):
        warmup_epochs = 3
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (config.IRN_EPOCHS - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    print(f"\n🚀 Starting IRNet training...")
    print(f"   Epochs: {config.IRN_EPOCHS}")
    print(f"   Gamma (displacement weight): {config.GAMMA}")
    print(f"   Early stopping: {config.EARLY_STOPPING_ENABLED}")
    if config.EARLY_STOPPING_ENABLED:
        print(f"   Patience: {config.IRN_EARLY_STOP_PATIENCE}, Min delta: {config.IRN_EARLY_STOP_MIN_DELTA}")
    
    history = {
        'total_loss': [],
        'pos_aff': [],
        'neg_aff': [],
        'dp_fg': [],
        'dp_bg': []
    }
    
    best_loss = float('inf')
    no_improve_epochs = 0
    
    for epoch in range(config.IRN_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.IRN_EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        metrics = train_irnet_epoch(
            model, loader, optimizer, device, epoch, gamma=config.GAMMA,
            edge_params=edge_params, dp_params=dp_params
        )
        
        # Update history
        for key in history.keys():
            history[key].append(metrics[key])
        
        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"   Total Loss: {metrics['total_loss']:.4f}")
        print(f"   Pos Aff Loss: {metrics['pos_aff']:.4f}")
        print(f"   Neg Aff Loss: {metrics['neg_aff']:.4f}")
        print(f"   DP FG Loss: {metrics['dp_fg']:.4f}")
        print(f"   DP BG Loss: {metrics['dp_bg']:.4f}")
        print(f"   Mask density (FG/BG/NEG): {metrics['pos_fg_density']:.4f} / {metrics['pos_bg_density']:.4f} / {metrics['neg_density']:.4f}")
        print(f"   Grad norm (edge/dp): {metrics['edge_grad_norm']:.4f} / {metrics['dp_grad_norm']:.4f}")
        print(f"   Valid batches: {metrics['valid_batches']}/{len(loader)}")
        print(f"   Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if metrics['total_loss'] < (best_loss - config.IRN_EARLY_STOP_MIN_DELTA):
            best_loss = metrics['total_loss']
            no_improve_epochs = 0
            save_path = os.path.join(config.OUTPUT_DIR, 'irnet_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metrics['total_loss'],
            }, save_path)
            print(f"   💾 Best model saved! (Loss: {best_loss:.4f})")
        else:
            no_improve_epochs += 1
            if config.EARLY_STOPPING_ENABLED:
                print(
                    f"   ⏳ No improvement for {no_improve_epochs}/{config.IRN_EARLY_STOP_PATIENCE} epochs"
                )
                if no_improve_epochs >= config.IRN_EARLY_STOP_PATIENCE:
                    print("\n🛑 Early stopping triggered on Stage 2+3 training")
                    break
    
    # Save final model
    final_path = os.path.join(config.OUTPUT_DIR, 'irnet_final.pth')
    torch.save({
        'epoch': config.IRN_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': metrics['total_loss'],
    }, final_path)
    print(f"\n💾 Final model saved: {final_path}")
    
    # Plot history
    plot_irnet_history(history)
    
    # Summary
    print("\n" + "=" * 60)
    print("IRNet TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Final Loss: {metrics['total_loss']:.4f}")
    
    return model


if __name__ == "__main__":
    model = main()
