"""
inference.py - Stage 4: Generate Pseudo Instance Segmentation Labels

Menggunakan trained CAM + IRNet untuk:
1. Extract CAM dari full image (patch-by-patch)
2. Extract boundary map (B) dan displacement field (D) dari IRNet
3. Synthesize pseudo instance labels 

Input: Full crack images (4032x3024)
Output: Pseudo segmentation masks
"""

import os
import csv
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

# Try importing DenseCRF
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    DENSECRF_AVAILABLE = True
except ImportError:
    DENSECRF_AVAILABLE = False
    print("⚠️  Warning: pydensecrf not available, skipping DenseCRF refinement")


def apply_densecrf(img_rgb, probs):
    """
    Apply DenseCRF refinement to probability map.
    
    Args:
        img_rgb: (H, W, 3) uint8 numpy array - original image for bilateral kernel
        probs: (H, W) float32 numpy array - probability map [0, 1] for crack class
    
    Returns:
        refined_probs: (H, W) float32 refined probability map [0, 1]
    """
    if not DENSECRF_AVAILABLE or not config.USE_DENSECRF:
        return probs
    
    H, W = probs.shape
    
    # Create 2-class probability map
    prob_bg = 1.0 - probs
    prob_fg = probs
    softmax_probs = np.stack([prob_bg, prob_fg], axis=0)  # (2, H, W)
    
    # Create DenseCRF model
    d = dcrf.DenseCRF2D(W, H, 2)
    
    # Set unary potentials (negative log probabilities)
    U = unary_from_softmax(softmax_probs)
    d.setUnaryEnergy(U)
    
    # Add pairwise potentials
    # Gaussian kernel (spatial smoothing)
    d.addPairwiseGaussian(
        sxy=config.DCRF_POS_XY_STD,
        compat=config.DCRF_POS_W,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )
    
    # Bilateral kernel (edge-aware smoothing using image)
    d.addPairwiseBilateral(
        sxy=config.DCRF_BI_XY_STD,
        srgb=config.DCRF_BI_RGB_STD,
        rgbim=img_rgb.astype(np.uint8),
        compat=config.DCRF_BI_W,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC
    )
    
    # Inference
    Q = d.inference(config.DCRF_ITER)
    Q = np.array(Q).reshape((2, H, W))
    
    # Return refined crack probability
    refined_probs = Q[1]  # Crack class
    
    return refined_probs


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
        B: (h, w) boundary probability map (2D numpy array)
        D: (h, w, 2) displacement field (3D numpy array)
    """
    irnet.eval()
    
    # Prepare batch: [original, h-flipped] for TTA
    t_orig = transform(patch_rgb)
    t_flip = t_orig.flip(-1)
    batch = torch.stack([t_orig, t_flip], dim=0).to(device)
    
    with torch.no_grad():
        # EdgeDisplacement.forward() returns (edge_out, dp_out)
        # edge_out: (h, w) tensor after sigmoid + TTA
        # dp_out: (2, h, w) tensor
        edge_out, dp_out = irnet(batch)
        
        # Convert to numpy, ensure correct shapes
        B = edge_out.cpu().numpy()  # Should be (h, w)
        D = dp_out.cpu().numpy()     # Should be (2, h, w)
        
        # Transpose D to (h, w, 2) for easier handling
        D = np.transpose(D, (1, 2, 0))  # (2, h, w) -> (h, w, 2)
        
        # Ensure B is 2D
        if len(B.shape) > 2:
            B = B.squeeze()
    
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


def synthesize_pseudo_label(cam, D, B, img_rgb=None,
                           beta=8, t_walk=256, bg_quantile=0.25, b_penalty=0.7):
    """
    Synthesize pseudo label via Hybrid CAM+IRN approach with DenseCRF refinement.
    
    Pipeline:
    1. Boundary-aware CAM refinement: Use IRN boundary map to suppress CAM at edges
    2. DenseCRF refinement: Edge-aware smoothing using original RGB image
    3. Thresholding and morphological operations
    4. Optional: Displacement-based instance clustering
    
    Args:
        cam: (H, W) CAM array (primary signal)
        D: (H, W, 2) displacement field for instance separation
        B: (H, W) boundary map for edge refinement
        img_rgb: (H, W, 3) original RGB image for DenseCRF (optional)
    
    Returns:
        pseudo_label: (H, W) binary segmentation mask
    """
    H, W = cam.shape
    
    # Work at full resolution for better accuracy
    # Enhance contrast with gamma correction
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_enhanced = np.power(cam_norm, 0.7)  # Boost crack regions
    
    # Debug: Print CAM statistics
    print(f"   [DEBUG synthesize] cam min/max: {cam.min():.4f}/{cam.max():.4f}")
    print(f"   [DEBUG synthesize] cam_enhanced min/max: {cam_enhanced.min():.4f}/{cam_enhanced.max():.4f}")
    print(f"   [DEBUG synthesize] CAM >0.3 ratio: {(cam_enhanced > 0.3).sum() / cam_enhanced.size:.4f}")
    
    # STAGE 1: Hybrid CAM + IRN Boundary Refinement
    if config.USE_BOUNDARY_REFINEMENT and B is not None:
        # Resize boundary map to match CAM resolution if needed
        if B.shape != (H, W):
            B_resized = cv2.resize(B, (W, H))
        else:
            B_resized = B

        # Restorative fusion: use IRN signal to recover weak/thin crack segments
        # near high-confidence CAM seeds, instead of globally suppressing CAM.
        B_norm = np.clip(B_resized, 0.0, 1.0)
        seed = (cam_enhanced >= config.IRN_RESTORE_SEED_THRESH).astype(np.uint8)

        if config.IRN_RESTORE_DILATION > 0:
            k = 2 * config.IRN_RESTORE_DILATION + 1
            seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            support_region = cv2.dilate(seed, seed_kernel, iterations=1).astype(np.float32)
        else:
            support_region = seed.astype(np.float32)

        irn_support = B_norm * support_region

        # Keep only IRN-support components connected to CAM seed anchors.
        support_bin = (irn_support >= config.IRN_RESTORE_SUPPORT_THRESH).astype(np.uint8)
        ak = 2 * config.IRN_RESTORE_CONNECT_DILATION + 1
        anchor_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ak, ak))
        seed_anchor = cv2.dilate(seed, anchor_kernel, iterations=1)

        n_cc, cc_map = cv2.connectedComponents(support_bin, connectivity=8)
        connected_support = np.zeros_like(support_bin, dtype=np.uint8)
        for cc_id in range(1, n_cc):
            cc_mask = (cc_map == cc_id)
            if np.any(seed_anchor[cc_mask] > 0):
                connected_support[cc_mask] = 1

        # Bridge tiny gaps in recovered support without noticeably thickening.
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        connected_support = cv2.morphologyEx(connected_support, cv2.MORPH_CLOSE, bridge_kernel, iterations=1)

        # Restore mostly low-confidence CAM areas so existing strong crack does not bloat.
        low_conf_mask = (cam_enhanced < config.IRN_RESTORE_LOW_CONF_THRESH).astype(np.float32)
        restoration = connected_support.astype(np.float32) * irn_support * low_conf_mask
        cam_hybrid = cam_enhanced + (restoration * config.IRN_RESTORE_GAIN * (1.0 - cam_enhanced))
        cam_hybrid = np.clip(cam_hybrid, 0.0, 1.0)
        
        print(f"   [DEBUG synthesize] Applied restorative CAM+IRN fusion")
        print(f"   [DEBUG synthesize] B min/max: {B_resized.min():.4f}/{B_resized.max():.4f}")
        print(f"   [DEBUG synthesize] recover support ratio: {connected_support.mean():.4f}")
        print(f"   [DEBUG synthesize] cam_hybrid min/max: {cam_hybrid.min():.4f}/{cam_hybrid.max():.4f}")
        
        cam_enhanced = cam_hybrid
    
    # STAGE 2: Apply DenseCRF refinement for edge-aware smoothing
    if img_rgb is not None and DENSECRF_AVAILABLE and config.USE_DENSECRF:
        print(f"   [DEBUG synthesize] Applying DenseCRF...")
        cam_enhanced = apply_densecrf(img_rgb, cam_enhanced)
        print(f"   [DEBUG synthesize] After DenseCRF min/max: {cam_enhanced.min():.4f}/{cam_enhanced.max():.4f}")
    
    # Light smoothing to preserve thin crack structure
    cam_smooth = cv2.GaussianBlur(cam_enhanced, (3, 3), 0)
    
    # Otsu thresholding with adjustment
    cam_uint8 = (cam_smooth * 255).astype(np.uint8)
    thresh_val, binary = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"   [DEBUG synthesize] Otsu threshold: {thresh_val}, pixels above: {(binary > 0).sum()}")
    
    # If Otsu is too conservative, lower threshold
    if thresh_val > 127:
        thresh_val = int(thresh_val * 0.6)
        _, binary = cv2.threshold(cam_uint8, thresh_val, 255, cv2.THRESH_BINARY)
        print(f"   [DEBUG synthesize] Adjusted threshold: {thresh_val}, pixels above: {(binary > 0).sum()}")
    
    # Gentle morphological operations to preserve thin crack structure
    if B is not None:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_iter = 1
    else:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        close_iter = 2
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))   # REDUCED from (3,3)

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=close_iter)

    # For CAM+IRN path, skip MORPH_OPEN to avoid breaking thin recovered links.
    if B is None:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    print(f"   [DEBUG synthesize] After morphology pixels: {(binary > 0).sum()}")
    
    # Remove small noise components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if B is not None:
        min_size = max(30, (H * W) // 40000)  # Keep thin recovered fragments for CAM+IRN
    else:
        min_size = max(80, (H * W) // 20000)  # Less aggressive filtering for thin cracks
    
    print(f"   [DEBUG synthesize] Connected components: {num_labels-1}, min_size: {min_size}")
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            binary[labels == i] = 0
    
    final_pixels = (binary > 0).sum()
    print(f"   [DEBUG synthesize] Final pixels after filtering: {final_pixels}")
    
    # STAGE 3 (Optional): Displacement-based instance clustering
    if config.USE_DISPLACEMENT_CLUSTERING and D is not None:
        print(f"   [DEBUG synthesize] Applying displacement-based clustering...")
        
        # Resize displacement field to match binary mask resolution if needed
        if D.shape[:2] != (H, W):
            D_resized = np.zeros((H, W, 2), dtype=np.float32)
            D_resized[:, :, 0] = cv2.resize(D[:, :, 0], (W, H))
            D_resized[:, :, 1] = cv2.resize(D[:, :, 1], (W, H))
        else:
            D_resized = D
        
        # Apply displacement field only to crack regions
        binary_clustered = apply_displacement_clustering(binary, D_resized)
        binary = binary_clustered
        
        print(f"   [DEBUG synthesize] After displacement clustering: {(binary > 0).sum()} pixels")
    
    return binary


def apply_displacement_clustering(binary_mask, displacement_field):
    """
    Apply displacement-based instance clustering to separate crack instances.
    
    Args:
        binary_mask: (H, W) binary mask of crack regions
        displacement_field: (H, W, 2) displacement vectors
    
    Returns:
        clustered_mask: (H, W) binary mask with refined instance boundaries
    """
    H, W = binary_mask.shape
    
    # Only process crack pixels
    crack_coords = np.column_stack(np.where(binary_mask > 0))
    
    if len(crack_coords) == 0:
        return binary_mask
    
    # Get displacement vectors for crack pixels
    displacements = displacement_field[crack_coords[:, 0], crack_coords[:, 1]]
    
    # Target positions = current positions + displacements
    target_positions = crack_coords + displacements
    
    # Cluster pixels by their target positions using connected components
    # Create a temporary target map
    target_map = np.zeros((H, W), dtype=np.int32)
    
    for i, (y, x) in enumerate(crack_coords):
        ty, tx = target_positions[i]
        ty = int(np.clip(ty, 0, H-1))
        tx = int(np.clip(tx, 0, W-1))
        target_map[y, x] = ty * W + tx
    
    # Group pixels pointing to similar targets
    # Use connected components on the crack mask, then filter by displacement coherence
    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    
    refined_mask = binary_mask.copy()
    
    for label_id in range(1, num_labels):
        component_mask = (labels == label_id)
        component_coords = np.column_stack(np.where(component_mask))
        
        if len(component_coords) < 10:
            continue
        
        # Check displacement coherence within this component
        component_targets = target_map[component_coords[:, 0], component_coords[:, 1]]
        
        # If targets are too scattered, this might be multiple instances
        unique_targets, counts = np.unique(component_targets, return_counts=True)
        
        # If displacement vectors are incoherent (pointing to many different places),
        # this could indicate boundary regions to remove
        coherence = counts.max() / len(component_coords)
        
        if coherence < 0.3:  # Low coherence → likely boundary/noise
            # Erode this component slightly
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded = cv2.erode(component_mask.astype(np.uint8), kernel, iterations=1)
            refined_mask[component_mask & (eroded == 0)] = 0
    
    return refined_mask


def _sliding_positions(length, patch_size, stride):
    """Generate sliding-window start indices with guaranteed edge coverage."""
    if length <= patch_size:
        return [0]

    positions = list(range(0, length - patch_size + 1, stride))
    last_start = length - patch_size
    if positions[-1] != last_start:
        positions.append(last_start)
    return positions


def _build_patch_weight(patch_size):
    """Build smooth 2D blending weights to reduce seam artifacts."""
    w = np.hanning(patch_size).astype(np.float32)
    if np.all(w == 0):
        w = np.ones((patch_size,), dtype=np.float32)
    weight = np.outer(w, w)
    weight = np.clip(weight, 1e-3, None)
    weight = weight / (weight.max() + 1e-8)
    return weight


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
        cam_only_full: (H, W) pseudo label from CAM only
        pseudo_full: (H, W) final pseudo label from CAM+IRN
    """
    H, W = img_rgb.shape[:2]
    
    # Accumulators
    cam_accum = np.zeros((H, W), dtype=np.float32)
    B_accum = np.zeros((H, W), dtype=np.float32)
    D_accum = np.zeros((H, W, 2), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    patch_weight = _build_patch_weight(patch_size)
    
    # Sliding window
    patches = []
    y_positions = _sliding_positions(H, patch_size, stride)
    x_positions = _sliding_positions(W, patch_size, stride)
    for y in y_positions:
        for x in x_positions:
            patches.append((y, x))
    
    print(f"   Processing {len(patches)} patches...")
    
    for y, x in tqdm(patches, desc="   Patches"):
        # Extract patch
        patch = img_rgb[y:y+patch_size, x:x+patch_size]
        
        # Extract CAM
        cam = extract_cam_for_patch(cam_net, patch, transform, device)
        
        # Extract boundary + displacement
        B, D = extract_edge_displacement(irnet, patch, transform, device)
        
        # Debug: check shapes
        if y == 0 and x == 0:  # First patch only
            print(f"   [DEBUG] cam shape: {cam.shape}")
            print(f"   [DEBUG] B shape: {B.shape}")
            print(f"   [DEBUG] D shape: {D.shape}")
        
        # Resize to patch size
        cam_resized = cv2.resize(cam, (patch_size, patch_size))
        B_resized = cv2.resize(B, (patch_size, patch_size))
        
        if y == 0 and x == 0:  # First patch only
            print(f"   [DEBUG] B_resized shape: {B_resized.shape}")
        
        # For displacement field, resize each channel separately
        D_resized = np.zeros((patch_size, patch_size, 2), dtype=np.float32)
        D_resized[:, :, 0] = cv2.resize(D[:, :, 0], (patch_size, patch_size))
        D_resized[:, :, 1] = cv2.resize(D[:, :, 1], (patch_size, patch_size))
        
        # Accumulate with smooth blending weights
        cam_accum[y:y+patch_size, x:x+patch_size] += cam_resized * patch_weight
        B_accum[y:y+patch_size, x:x+patch_size] += B_resized * patch_weight
        D_accum[y:y+patch_size, x:x+patch_size] += D_resized * patch_weight[:, :, None]
        count[y:y+patch_size, x:x+patch_size] += patch_weight
    
    # Average
    count[count == 0] = 1
    cam_full = cam_accum / count
    B_full = B_accum / count
    D_full = D_accum / count[:, :, None]
    
    print(f"   [DEBUG process] cam_full min/max: {cam_full.min():.4f}/{cam_full.max():.4f}")
    print(f"   [DEBUG process] cam_full mean: {cam_full.mean():.4f}")
    print(f"   [DEBUG process] cam_full >0.1 ratio: {(cam_full > 0.1).sum() / cam_full.size:.4f}")
    
    print("   Synthesizing pseudo label...")
    
    # Downsample untuk synthesis (memory efficient)
    # Increased downsampling for faster processing
    ds_factor = 2  # Use 2 for higher quality synthesis (was 8)
    h_ds = H // ds_factor
    w_ds = W // ds_factor
    
    cam_ds = cv2.resize(cam_full, (w_ds, h_ds))
    B_ds = cv2.resize(B_full, (w_ds, h_ds))
    
    # Properly resize displacement field (each channel separately)
    D_ds = np.zeros((h_ds, w_ds, 2), dtype=np.float32)
    D_ds[:, :, 0] = cv2.resize(D_full[:, :, 0], (w_ds, h_ds))
    D_ds[:, :, 1] = cv2.resize(D_full[:, :, 1], (w_ds, h_ds))
    
    img_rgb_ds = cv2.resize(img_rgb, (w_ds, h_ds))  # Downsample RGB for DenseCRF
    
    # Synthesize CAM-only label
    cam_only_full = synthesize_pseudo_label(
        cam_ds, None, None,
        img_rgb=img_rgb_ds,
        beta=config.BETA,
        t_walk=config.T_WALK,
        bg_quantile=config.BG_QUANTILE,
        b_penalty=config.B_PENALTY
    )

    # Synthesize CAM+IRN pseudo label with DenseCRF refinement
    pseudo_full = synthesize_pseudo_label(
        cam_ds, D_ds, B_ds,
        img_rgb=img_rgb_ds,  # Pass RGB image for DenseCRF
        beta=config.BETA,
        t_walk=config.T_WALK,
        bg_quantile=config.BG_QUANTILE,
        b_penalty=config.B_PENALTY
    )
    
    # Resize back to full size
    cam_only_full = cv2.resize(cam_only_full, (W, H), interpolation=cv2.INTER_NEAREST)
    pseudo_full = cv2.resize(pseudo_full, (W, H), interpolation=cv2.INTER_NEAREST)
    
    return cam_full, B_full, D_full, cam_only_full, pseudo_full


def visualize_results(img_rgb, cam, B, D, cam_only, pseudo, gt=None, save_path=None):
    """Visualize inference results"""
    ncols = 8 if gt is None else 9
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
    
    # CAM-only label
    axes[4].imshow(cam_only, cmap='gray')
    axes[4].set_title('CAM Only Label')
    axes[4].axis('off')

    # CAM+IRN pseudo label
    axes[5].imshow(pseudo, cmap='gray')
    axes[5].set_title('CAM + IRN Label')
    axes[5].axis('off')
    
    # CAM-only overlay
    overlay_cam = img_rgb.copy()
    overlay_cam[cam_only > 0] = overlay_cam[cam_only > 0] * 0.5 + np.array([0, 255, 255]) * 0.5
    axes[6].imshow(overlay_cam.astype(np.uint8))
    axes[6].set_title('Overlay CAM Only')
    axes[6].axis('off')

    # CAM+IRN overlay
    overlay_irn = img_rgb.copy()
    overlay_irn[pseudo > 0] = overlay_irn[pseudo > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    axes[7].imshow(overlay_irn.astype(np.uint8))
    axes[7].set_title('Overlay CAM + IRN')
    axes[7].axis('off')
    
    # Ground truth (if available)
    if gt is not None:
        axes[8].imshow(gt, cmap='gray')
        axes[8].set_title('Ground Truth')
        axes[8].axis('off')
    
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


def save_and_visualize_overall_metrics(metrics_records, output_dir, prefix='cam_irn'):
    """Save per-image metrics and create overall evaluation visualization."""
    if not metrics_records:
        return None

    os.makedirs(output_dir, exist_ok=True)

    metric_names = ['IoU', 'Precision', 'Recall', 'F1']
    values = np.array([[record[m] for m in metric_names] for record in metrics_records], dtype=np.float32)

    means = values.mean(axis=0)
    stds = values.std(axis=0)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)

    # Save per-image metrics CSV
    per_image_csv = os.path.join(output_dir, f'evaluation_metrics_per_image_{prefix}.csv')
    with open(per_image_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename'] + metric_names)
        for record in metrics_records:
            writer.writerow([record['filename']] + [f"{record[m]:.6f}" for m in metric_names])

    # Save summary CSV
    summary_csv = os.path.join(output_dir, f'evaluation_metrics_summary_{prefix}.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'mean', 'std', 'min', 'max'])
        for i, name in enumerate(metric_names):
            writer.writerow([
                name,
                f"{means[i]:.6f}",
                f"{stds[i]:.6f}",
                f"{mins[i]:.6f}",
                f"{maxs[i]:.6f}"
            ])

    # Create visualization figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean ± std
    x = np.arange(len(metric_names))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    axes[0].bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_names)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Overall Metrics (Mean ± Std)')
    axes[0].grid(True, axis='y', alpha=0.3)

    for i, v in enumerate(means):
        axes[0].text(i, min(v + 0.03, 0.98), f"{v:.3f}", ha='center', va='bottom', fontsize=9)

    # Right: per-image distribution
    axes[1].boxplot([values[:, i] for i in range(len(metric_names))], tick_labels=metric_names, showmeans=True)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel('Score')
    axes[1].set_title('Per-Image Metric Distribution')
    axes[1].grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'{prefix.upper()} Evaluation on {len(metrics_records)} Images', fontsize=13)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'evaluation_metrics_overall_{prefix}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        'plot_path': plot_path,
        'per_image_csv': per_image_csv,
        'summary_csv': summary_csv,
        'means': {name: float(means[i]) for i, name in enumerate(metric_names)},
        'stds': {name: float(stds[i]) for i, name in enumerate(metric_names)}
    }


def save_comparison_metrics_plot(cam_summary, irn_summary, output_dir):
    """Create side-by-side comparison plot for CAM-only vs CAM+IRN metrics."""
    metric_names = ['IoU', 'Precision', 'Recall', 'F1']
    cam_means = [cam_summary['means'][m] for m in metric_names]
    irn_means = [irn_summary['means'][m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.36

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.bar(x - width / 2, cam_means, width, label='CAM Only', color='#17becf')
    ax.bar(x + width / 2, irn_means, width, label='CAM + IRN', color='#d62728')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('CAM-only vs CAM+IRN Mean Metrics')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()

    for i, v in enumerate(cam_means):
        ax.text(i - width / 2, min(v + 0.02, 0.98), f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(irn_means):
        ax.text(i + width / 2, min(v + 0.02, 0.98), f"{v:.3f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    cmp_path = os.path.join(output_dir, 'evaluation_metrics_comparison_cam_vs_cam_irn.png')
    plt.savefig(cmp_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return cmp_path


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
    checkpoint = torch.load(cam_path, map_location=device, weights_only=False)
    cam_net.load_state_dict(checkpoint['model_state_dict'])
    cam_net.eval()
    print(f"   ✅ CAM network loaded")
    
    # IRNet - load from AffinityDisplacementLoss checkpoint
    from path_index import PathIndex
    
    # IMPORTANT: Must match the size used during training!
    # IRNet training uses 256×256 patches → 64×64 feature maps
    irn_patch_size = 256
    irn_feat_h = irn_patch_size // 4  # 64
    irn_feat_w = irn_patch_size // 4  # 64
    
    # First create AffinityDisplacementLoss to load the checkpoint
    path_idx = PathIndex(feat_h=irn_feat_h, feat_w=irn_feat_w, radius=config.RADIUS)
    irnet_temp = AffinityDisplacementLoss(path_idx).to(device)
    
    irnet_path = os.path.join(config.OUTPUT_DIR, 'irnet_best.pth')
    checkpoint = torch.load(irnet_path, map_location=device, weights_only=False)
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
    
    all_metrics_cam_only = []
    all_metrics_cam_irn = []
    
    for idx in range(len(dataset)):
        img_rgb, gt, fname = dataset[idx]
        
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(dataset)}] {fname}")
        print(f"{'='*60}")
        print(f"   Image size: {img_rgb.shape[0]}x{img_rgb.shape[1]}")
        
        # Process
        cam, B, D, cam_only, pseudo = process_full_image(
            img_rgb, cam_net, irnet, transform, device,
            patch_size=config.INFERENCE_PATCH_SIZE,
            stride=config.INFERENCE_STRIDE
        )
        
        # Save CAM-only and CAM+IRN pseudo labels
        cam_only_path = os.path.join(pseudo_dir, fname.replace('.jpg', '_cam_only.png'))
        pseudo_path = os.path.join(pseudo_dir, fname.replace('.jpg', '_cam_irn_pseudo.png'))
        cv2.imwrite(cam_only_path, cam_only)
        cv2.imwrite(pseudo_path, pseudo)
        print(f"   ✅ CAM-only label saved: {cam_only_path}")
        print(f"   ✅ CAM+IRN label saved: {pseudo_path}")
        
        # Visualize
        if config.SAVE_VISUALIZATIONS:
            vis_path = os.path.join(vis_dir, fname.replace('.jpg', '_vis.png'))
            visualize_results(img_rgb, cam, B, D, cam_only, pseudo, gt, vis_path)
        
        # Compute metrics if GT available
        if gt is not None and config.COMPUTE_METRICS:
            metrics_cam = compute_metrics(cam_only, gt)
            metrics_irn = compute_metrics(pseudo, gt)
            all_metrics_cam_only.append({
                'filename': fname,
                **metrics_cam
            })
            all_metrics_cam_irn.append({
                'filename': fname,
                **metrics_irn
            })
            
            print(f"\n   📊 Metrics:")
            print(f"      CAM-only  IoU/Prec/Rec/F1: {metrics_cam['IoU']:.4f} / {metrics_cam['Precision']:.4f} / {metrics_cam['Recall']:.4f} / {metrics_cam['F1']:.4f}")
            print(f"      CAM+IRN   IoU/Prec/Rec/F1: {metrics_irn['IoU']:.4f} / {metrics_irn['Precision']:.4f} / {metrics_irn['Recall']:.4f} / {metrics_irn['F1']:.4f}")
    
    # Average metrics
    if all_metrics_cam_irn:
        print(f"\n{'='*60}")
        print("AVERAGE METRICS")
        print(f"{'='*60}")
        
        avg_metrics_cam = {
            key: np.mean([m[key] for m in all_metrics_cam_only])
            for key in ['IoU', 'Precision', 'Recall', 'F1']
        }
        avg_metrics_irn = {
            key: np.mean([m[key] for m in all_metrics_cam_irn])
            for key in ['IoU', 'Precision', 'Recall', 'F1']
        }

        print("   CAM-only:")
        for key, val in avg_metrics_cam.items():
            print(f"      {key}: {val:.4f}")

        print("   CAM+IRN:")
        for key, val in avg_metrics_irn.items():
            print(f"      {key}: {val:.4f}")

        cam_summary = save_and_visualize_overall_metrics(
            all_metrics_cam_only, config.OUTPUT_DIR, prefix='cam_only'
        )
        irn_summary = save_and_visualize_overall_metrics(
            all_metrics_cam_irn, config.OUTPUT_DIR, prefix='cam_irn'
        )

        if cam_summary is not None and irn_summary is not None:
            cmp_path = save_comparison_metrics_plot(cam_summary, irn_summary, config.OUTPUT_DIR)
            print(f"\n📊 CAM-only overall plot: {cam_summary['plot_path']}")
            print(f"📊 CAM+IRN overall plot: {irn_summary['plot_path']}")
            print(f"📊 Comparison plot: {cmp_path}")
            print(f"🧾 CAM-only per-image CSV: {cam_summary['per_image_csv']}")
            print(f"🧾 CAM+IRN per-image CSV: {irn_summary['per_image_csv']}")
    
    print(f"\n✅ All done!")
    print(f"   Pseudo labels saved in: {pseudo_dir}")
    print(f"   Visualizations saved in: {vis_dir}")


if __name__ == "__main__":
    main()