"""
inference.py - Stage 4: Generate Pseudo Instance Segmentation Labels

Menggunakan trained CAM + IRNet untuk:
1. Extract CAM dari full image (patch-by-patch)
2. Extract boundary map (B) dan displacement field (D) dari IRNet
3. Synthesize pseudo instance labels menggunakan random walk (Sec 5)

Input: Full crack images (4032x3024)
Output: Pseudo segmentation masks
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags
from scipy.ndimage import label

import config
from resnet50_cam import CAM as CamExtractor
from resnet50_irn import EdgeDisplacement, AffinityDisplacementLoss
from dataset import get_inference_transform, CrackFullImageDataset


def extract_cam_for_patch(cam_net, patch_rgb, transform, device):
    """
    Extract CAM untuk single patch.
    
    Args:
        cam_net: Trained CAM network
        patch_rgb: (H, W, 3) numpy array
        transform: Transform function
        device: torch device
    
    Returns:
        cam: (h, w) numpy array, normalized to [0, 1]
    """
    cam_net.eval()
    
    # Prepare batch: [original, h-flipped]
    t_orig = transform(patch_rgb)
    t_flip = t_orig.flip(-1)
    batch = torch.stack([t_orig, t_flip], dim=0).to(device)
    
    with torch.no_grad():
        x = cam_net.stage1(batch)
        x = cam_net.stage2(x)
        x = cam_net.stage3(x)
        x = cam_net.stage4(x)
        
        # CAM = φ^T * f(x)
        x = F.conv2d(x, cam_net.classifier.weight)
        x = F.relu(x)
        
        # TTA: average original + flipped
        cam_2d = x[0] + x[1].flip(-1)  # (NUM_CLASSES, h, w)
    
    # Take crack class (index 1)
    cam = cam_2d[1].cpu().numpy()
    
    # Normalize
    cam = cam / (cam.max() + 1e-8)
    
    return cam


def extract_edge_displacement(irnet, patch_rgb, transform, device):
    """
    Extract boundary map dan displacement field untuk patch.
    
    Args:
        irnet: Trained EdgeDisplacement model
        patch_rgb: (H, W, 3) numpy array
        transform: Transform function
        device: torch device
    
    Returns:
        B: (h, w) boundary probability map
        D: (h, w, 2) displacement field (dy, dx)
    """
    irnet.eval()
    
    # Prepare batch: [original, h-flipped] for TTA
    t_orig = transform(patch_rgb)
    t_flip = t_orig.flip(-1)
    batch = torch.stack([t_orig, t_flip], dim=0).to(device)
    
    with torch.no_grad():
        # EdgeDisplacement.forward() returns (edge_out, dp_out)
        # edge_out: (h, w) after sigmoid + TTA
        # dp_out: (2, h, w)
        edge_out, dp_out = irnet(batch)
        
        # Convert to numpy
        B = edge_out.cpu().numpy()  # (h, w)
        D = dp_out.permute(1, 2, 0).cpu().numpy()  # (h, w, 2)
    
    return B, D


def refine_displacement(D, n_iter=100, radius=1.0):
    """
    Refine displacement field via iterative averaging (mean-shift like).
    
    Args:
        D: (h, w, 2) displacement field
        n_iter: number of iterations
        radius: neighbor radius
    
    Returns:
        D_refined: (h, w, 2) refined displacement
    """
    h, w = D.shape[:2]
    D_refined = D.copy()
    
    for _ in range(n_iter):
        D_new = np.zeros_like(D_refined)
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                
                # Shift field
                if dy < 0:
                    y_slice = slice(0, h+dy)
                    y_slice_new = slice(-dy, h)
                elif dy > 0:
                    y_slice = slice(dy, h)
                    y_slice_new = slice(0, h-dy)
                else:
                    y_slice = slice(0, h)
                    y_slice_new = slice(0, h)
                
                if dx < 0:
                    x_slice = slice(0, w+dx)
                    x_slice_new = slice(-dx, w)
                elif dx > 0:
                    x_slice = slice(dx, w)
                    x_slice_new = slice(0, w-dx)
                else:
                    x_slice = slice(0, w)
                    x_slice_new = slice(0, w)
                
                D_new[y_slice_new, x_slice_new] += D_refined[y_slice, x_slice]
        
        D_refined = D_new / 8.0  # Average over 8 neighbors
    
    return D_refined


def displacement_to_instance_map(D):
    """
    Convert displacement field to instance map via clustering.
    
    Args:
        D: (h, w, 2) displacement field
    
    Returns:
        instance_map: (h, w) int array with instance IDs
    """
    h, w = D.shape[:2]
    
    # Round displacements to integer pixel coordinates
    D_int = np.round(D).astype(np.int32)
    
    # Target positions = current positions + displacements
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    target_y = np.clip(yy + D_int[:, :, 0], 0, h-1)
    target_x = np.clip(xx + D_int[:, :, 1], 0, w-1)
    
    # Create target map
    target_map = target_y * w + target_x
    
    # Group pixels pointing to same target
    unique_targets = np.unique(target_map)
    instance_map = np.zeros((h, w), dtype=np.int32)
    
    for inst_id, target in enumerate(unique_targets, start=1):
        mask = (target_map == target)
        instance_map[mask] = inst_id
    
    return instance_map


def synthesize_pseudo_label(cam, D, B, 
                           beta=8, t_walk=256, bg_quantile=0.25, b_penalty=0.7):
    """
    Synthesize pseudo instance label via random walk (Sec 5 in paper).
    
    Args:
        cam: (H, W) CAM array
        D: (h, w, 2) displacement field  
        B: (h, w) boundary map
        beta: affinity power
        t_walk: random walk iterations
        bg_quantile: background threshold quantile
        b_penalty: boundary penalty (1-B)
    
    Returns:
        pseudo_label: (H, W) binary segmentation mask
    """
    H, W = cam.shape
    h, w = B.shape
    
    # Resize CAM to feature map size
    cam_feat = cv2.resize(cam, (w, h))
    
    # Get instance map from displacement
    D_refined = refine_displacement(D, n_iter=100)
    instance_map = displacement_to_instance_map(D_refined)
    
    # Get unique instances
    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids > 0]  # Remove background
    
    if len(instance_ids) == 0:
        # No instances found, threshold CAM directly
        pseudo_label = (cam > 0.5).astype(np.uint8) * 255
        return pseudo_label
    
    # Build affinity matrix (simplified version)
    N = h * w
    B_flat = B.flatten()
    
    # Create sparse affinity matrix
    rows, cols, vals = [], [], []
    
    # 8-connected neighbors
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            
            # Valid source region
            sy_start = max(0, -dy)
            sy_end = min(h, h - dy)
            sx_start = max(0, -dx)
            sx_end = min(w, w - dx)
            
            for sy in range(sy_start, sy_end):
                for sx in range(sx_start, sx_end):
                    src_idx = sy * w + sx
                    dst_idx = (sy + dy) * w + (sx + dx)
                    
                    # Affinity: 1 - max boundary between src and dst
                    a_ij = 1.0 - max(B_flat[src_idx], B_flat[dst_idx])
                    a_ij = np.clip(a_ij, 1e-7, 1.0)
                    a_beta = a_ij ** beta
                    
                    rows.append(src_idx)
                    cols.append(dst_idx)
                    vals.append(a_beta)
    
    # Build sparse affinity matrix
    A = csr_matrix((vals, (rows, cols)), shape=(N, N))
    
    # Row-normalize to get transition matrix
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    T = diags(1.0 / row_sums) @ A
    
    # Random walk for each instance
    best_scores = np.zeros(N, dtype=np.float32)
    best_labels = np.zeros(N, dtype=np.int32)
    
    B_penalty = np.maximum(1 - B_flat, b_penalty)
    
    for k in instance_ids:
        # Seed for instance k
        mask_k = (instance_map.flatten() == k).astype(np.float32)
        seed = cam_feat.flatten() * mask_k * B_penalty
        
        # Random walk
        vec = seed.copy()
        for _ in range(t_walk):
            vec = T.dot(vec)
        
        # Update best labels
        better = vec > best_scores
        best_scores[better] = vec[better]
        best_labels[better] = k
    
    # Threshold low scores as background
    pos_scores = best_scores[best_scores > 0]
    if len(pos_scores) > 0:
        thresh = np.quantile(pos_scores, bg_quantile)
        best_labels[best_scores < thresh] = 0
    
    # Reshape to image
    label_feat = best_labels.reshape(h, w)
    
    # Binary mask: any instance > 0
    pseudo_binary = (label_feat > 0).astype(np.uint8) * 255
    
    # Resize to original image size
    pseudo_label = cv2.resize(pseudo_binary, (W, H), interpolation=cv2.INTER_NEAREST)
    
    return pseudo_label


def process_full_image(img_rgb, cam_net, irnet, transform, device,
                      patch_size=512, stride=256):
    """
    Process full image dengan sliding window approach.
    
    Args:
        img_rgb: (H, W, 3) numpy array
        cam_net: Trained CAM network
        irnet: Trained IRNet network
        transform: Transform function
        device: torch device
        patch_size: Patch size untuk processing
        stride: Stride untuk overlap
    
    Returns:
        cam_full: (H, W) accumulated CAM
        B_full: (H, W) accumulated boundary map
        D_full: (H, W, 2) accumulated displacement field
        pseudo_full: (H, W) final pseudo label
    """
    H, W = img_rgb.shape[:2]
    
    # Accumulators
    cam_accum = np.zeros((H, W), dtype=np.float32)
    B_accum = np.zeros((H, W), dtype=np.float32)
    D_accum = np.zeros((H, W, 2), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    
    # Sliding window
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patches.append((y, x))
    
    print(f"   Processing {len(patches)} patches...")
    
    for y, x in tqdm(patches, desc="   Patches"):
        # Extract patch
        patch = img_rgb[y:y+patch_size, x:x+patch_size]
        
        # Extract CAM
        cam = extract_cam_for_patch(cam_net, patch, transform, device)
        
        # Extract boundary + displacement
        B, D = extract_edge_displacement(irnet, patch, transform, device)
        
        # Resize to patch size
        cam_resized = cv2.resize(cam, (patch_size, patch_size))
        B_resized = cv2.resize(B, (patch_size, patch_size))
        
        # For displacement field, resize each channel separately
        D_resized = np.zeros((patch_size, patch_size, 2), dtype=np.float32)
        D_resized[:, :, 0] = cv2.resize(D[:, :, 0], (patch_size, patch_size))
        D_resized[:, :, 1] = cv2.resize(D[:, :, 1], (patch_size, patch_size))
        
        # Accumulate
        cam_accum[y:y+patch_size, x:x+patch_size] += cam_resized
        B_accum[y:y+patch_size, x:x+patch_size] += B_resized
        D_accum[y:y+patch_size, x:x+patch_size] += D_resized
        count[y:y+patch_size, x:x+patch_size] += 1
    
    # Average
    count[count == 0] = 1
    cam_full = cam_accum / count
    B_full = B_accum / count
    D_full = D_accum / count[:, :, None]
    
    # Downsample untuk synthesis (memory efficient)
    ds_factor = 4
    h_ds = H // ds_factor
    w_ds = W // ds_factor
    
    cam_ds = cv2.resize(cam_full, (w_ds, h_ds))
    B_ds = cv2.resize(B_full, (w_ds, h_ds))
    D_ds = cv2.resize(D_full, (w_ds, h_ds))
    
    # Synthesize pseudo label
    pseudo_full = synthesize_pseudo_label(
        cam_ds, D_ds, B_ds,
        beta=config.BETA,
        t_walk=config.T_WALK,
        bg_quantile=config.BG_QUANTILE,
        b_penalty=config.B_PENALTY
    )
    
    # Resize back to full size
    pseudo_full = cv2.resize(pseudo_full, (W, H), interpolation=cv2.INTER_NEAREST)
    
    return cam_full, B_full, D_full, pseudo_full


def visualize_results(img_rgb, cam, B, D, pseudo, gt=None, save_path=None):
    """Visualize inference results"""
    ncols = 6 if gt is None else 7
    fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
    
    # Original image
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # CAM
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('CAM')
    axes[1].axis('off')
    
    # Boundary map
    axes[2].imshow(B, cmap='hot')
    axes[2].set_title('Boundary Map')
    axes[2].axis('off')
    
    # Displacement magnitude
    D_mag = np.linalg.norm(D, axis=2)
    axes[3].imshow(D_mag, cmap='viridis')
    axes[3].set_title('Displacement Magnitude')
    axes[3].axis('off')
    
    # Pseudo label
    axes[4].imshow(pseudo, cmap='gray')
    axes[4].set_title('Pseudo Label')
    axes[4].axis('off')
    
    # Overlay
    overlay = img_rgb.copy()
    overlay[pseudo > 0] = overlay[pseudo > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    axes[5].imshow(overlay.astype(np.uint8))
    axes[5].set_title('Overlay')
    axes[5].axis('off')
    
    # Ground truth (if available)
    if gt is not None:
        axes[6].imshow(gt, cmap='gray')
        axes[6].set_title('Ground Truth')
        axes[6].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Visualization saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_metrics(pred, gt):
    """Compute IoU, Precision, Recall, F1"""
    # Binarize
    pred_bin = (pred > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)
    
    # Metrics
    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    
    iou = intersection / (union + 1e-8)
    
    tp = intersection
    fp = (pred_bin & ~gt_bin).sum()
    fn = (~pred_bin & gt_bin).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'IoU': iou,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }


def main():
    print("=" * 60)
    print("STAGE 4: Pseudo Label Synthesis")
    print("=" * 60)
    
    device = config.DEVICE
    print(f"\n🖥️  Device: {device}")
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    pseudo_dir = os.path.join(config.OUTPUT_DIR, 'pseudo_labels')
    vis_dir = os.path.join(config.OUTPUT_DIR, 'visualizations')
    os.makedirs(pseudo_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load models
    print(f"\n📥 Loading models...")
    
    # CAM network
    cam_net = CamExtractor().to(device)
    cam_path = os.path.join(config.OUTPUT_DIR, 'cam_net_best.pth')
    checkpoint = torch.load(cam_path, map_location=device)
    cam_net.load_state_dict(checkpoint['model_state_dict'])
    cam_net.eval()
    print(f"   ✅ CAM network loaded")
    
    # IRNet - load from AffinityDisplacementLoss checkpoint
    from path_index import PathIndex
    
    # First create AffinityDisplacementLoss to load the checkpoint
    path_idx = PathIndex(feat_h=config.FEAT_H, feat_w=config.FEAT_W, radius=config.RADIUS)
    irnet_temp = AffinityDisplacementLoss(path_idx).to(device)
    
    irnet_path = os.path.join(config.OUTPUT_DIR, 'irnet_best.pth')
    checkpoint = torch.load(irnet_path, map_location=device)
    irnet_temp.load_state_dict(checkpoint['model_state_dict'])
    
    # Now create EdgeDisplacement and copy the base Net parameters
    irnet = EdgeDisplacement(crop_size=config.INFERENCE_PATCH_SIZE).to(device)
    
    # Copy state dict from base Net (exclude path_indices and disp_target)
    state_dict = checkpoint['model_state_dict']
    # Filter out the path_indices and disp_target buffers
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                          if not k.startswith('path_indices') and k != 'disp_target'}
    
    irnet.load_state_dict(filtered_state_dict, strict=False)
    irnet.eval()
    
    del irnet_temp, path_idx  # Clean up
    print(f"   ✅ IRNet loaded")
    
    # Dataset
    dataset = CrackFullImageDataset(
        img_dir=config.IMG_DIR,
        mask_dir=config.MASK_DIR
    )
    
    transform = get_inference_transform()
    
    # Process images
    print(f"\n🎨 Processing images...")
    
    all_metrics = []
    
    for idx in range(len(dataset)):
        img_rgb, gt, fname = dataset[idx]
        
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(dataset)}] {fname}")
        print(f"{'='*60}")
        print(f"   Image size: {img_rgb.shape[0]}x{img_rgb.shape[1]}")
        
        # Process
        cam, B, D, pseudo = process_full_image(
            img_rgb, cam_net, irnet, transform, device,
            patch_size=config.INFERENCE_PATCH_SIZE,
            stride=config.INFERENCE_STRIDE
        )
        
        # Save pseudo label
        pseudo_path = os.path.join(pseudo_dir, fname.replace('.jpg', '_pseudo.png'))
        cv2.imwrite(pseudo_path, pseudo)
        print(f"   ✅ Pseudo label saved: {pseudo_path}")
        
        # Visualize
        if config.SAVE_VISUALIZATIONS:
            vis_path = os.path.join(vis_dir, fname.replace('.jpg', '_vis.png'))
            visualize_results(img_rgb, cam, B, D, pseudo, gt, vis_path)
        
        # Compute metrics if GT available
        if gt is not None and config.COMPUTE_METRICS:
            metrics = compute_metrics(pseudo, gt)
            all_metrics.append(metrics)
            
            print(f"\n   📊 Metrics:")
            print(f"      IoU: {metrics['IoU']:.4f}")
            print(f"      Precision: {metrics['Precision']:.4f}")
            print(f"      Recall: {metrics['Recall']:.4f}")
            print(f"      F1: {metrics['F1']:.4f}")
    
    # Average metrics
    if all_metrics:
        print(f"\n{'='*60}")
        print("AVERAGE METRICS")
        print(f"{'='*60}")
        
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        for key, val in avg_metrics.items():
            print(f"   {key}: {val:.4f}")
    
    print(f"\n✅ All done!")
    print(f"   Pseudo labels saved in: {pseudo_dir}")
    print(f"   Visualizations saved in: {vis_dir}")


if __name__ == "__main__":
    main()