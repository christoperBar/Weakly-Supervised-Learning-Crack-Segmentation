"""
utils.py - Utility functions untuk crack segmentation project

Includes:
- Visualization helpers
- Metrics computation
- Image processing utilities
- Model checkpoint management
"""

import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import config


def create_visualization_colormap():
    """Create custom colormap untuk crack visualization"""
    colors = ['black', 'red', 'yellow']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('crack', colors, N=n_bins)
    return cmap


def overlay_mask_on_image(image, mask, alpha=0.5, color=(255, 0, 0)):
    """
    Overlay binary mask on image.
    
    Args:
        image: (H, W, 3) RGB image
        mask: (H, W) binary mask (0 or 255)
        alpha: transparency
        color: RGB color for mask
    
    Returns:
        overlay: (H, W, 3) overlayed image
    """
    overlay = image.copy()
    mask_bool = mask > 127
    
    # Apply color to mask regions
    overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
    
    return overlay.astype(np.uint8)


def compute_iou(pred, gt, threshold=127):
    """
    Compute IoU between prediction and ground truth.
    
    Args:
        pred: (H, W) prediction mask
        gt: (H, W) ground truth mask
        threshold: binarization threshold
    
    Returns:
        iou: float
    """
    pred_bin = (pred > threshold).astype(np.uint8)
    gt_bin = (gt > threshold).astype(np.uint8)
    
    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_precision_recall_f1(pred, gt, threshold=127):
    """
    Compute Precision, Recall, F1.
    
    Args:
        pred: (H, W) prediction
        gt: (H, W) ground truth
        threshold: binarization threshold
    
    Returns:
        dict with precision, recall, f1
    """
    pred_bin = (pred > threshold).astype(np.uint8)
    gt_bin = (gt > threshold).astype(np.uint8)
    
    tp = (pred_bin & gt_bin).sum()
    fp = (pred_bin & ~gt_bin).sum()
    fn = (~pred_bin & gt_bin).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_dice(pred, gt, threshold=127):
    """
    Compute Dice coefficient.
    
    Args:
        pred: (H, W) prediction
        gt: (H, W) ground truth
        threshold: binarization threshold
    
    Returns:
        dice: float
    """
    pred_bin = (pred > threshold).astype(np.uint8)
    gt_bin = (gt > threshold).astype(np.uint8)
    
    intersection = (pred_bin & gt_bin).sum()
    
    dice = 2 * intersection / (pred_bin.sum() + gt_bin.sum() + 1e-8)
    
    return dice


def compute_all_metrics(pred, gt, threshold=127):
    """
    Compute all metrics at once.
    
    Returns:
        dict with IoU, Precision, Recall, F1, Dice
    """
    iou = compute_iou(pred, gt, threshold)
    prf = compute_precision_recall_f1(pred, gt, threshold)
    dice = compute_dice(pred, gt, threshold)
    
    return {
        'IoU': iou,
        'Precision': prf['precision'],
        'Recall': prf['recall'],
        'F1': prf['f1'],
        'Dice': dice
    }


def visualize_comparison(image, pred, gt, save_path=None):
    """
    Visualize comparison: image, prediction, ground truth, overlay.
    
    Args:
        image: (H, W, 3) RGB image
        pred: (H, W) prediction mask
        gt: (H, W) ground truth mask
        save_path: path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Prediction
    axes[0, 1].imshow(pred, cmap='gray')
    axes[0, 1].set_title('Prediction')
    axes[0, 1].axis('off')
    
    # Ground truth
    axes[1, 0].imshow(gt, cmap='gray')
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')
    
    # Overlay comparison
    # Green = True Positive, Red = False Positive, Blue = False Negative
    overlay = np.zeros((*pred.shape, 3), dtype=np.uint8)
    pred_bin = pred > 127
    gt_bin = gt > 127
    
    # True Positive - Green
    overlay[pred_bin & gt_bin] = [0, 255, 0]
    # False Positive - Red
    overlay[pred_bin & ~gt_bin] = [255, 0, 0]
    # False Negative - Blue
    overlay[~pred_bin & gt_bin] = [0, 0, 255]
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('TP(Green) FP(Red) FN(Blue)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def resize_maintaining_aspect(image, target_size):
    """
    Resize image maintaining aspect ratio.
    
    Args:
        image: (H, W, ...) numpy array
        target_size: (target_h, target_w) or int (square)
    
    Returns:
        resized: resized image
        scale: scale factor used
    """
    h, w = image.shape[:2]
    
    if isinstance(target_size, int):
        target_h = target_w = target_size
    else:
        target_h, target_w = target_size
    
    # Compute scale
    scale_h = target_h / h
    scale_w = target_w / w
    scale = min(scale_h, scale_w)
    
    # New size
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h))
    
    return resized, scale


def pad_to_size(image, target_size, pad_value=0):
    """
    Pad image to target size.
    
    Args:
        image: (H, W, ...) numpy array
        target_size: (target_h, target_w)
        pad_value: padding value
    
    Returns:
        padded: padded image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    pad_h = target_h - h
    pad_w = target_w - w
    
    if len(image.shape) == 2:
        padded = np.pad(image, 
                       ((0, pad_h), (0, pad_w)),
                       mode='constant',
                       constant_values=pad_value)
    else:
        padded = np.pad(image,
                       ((0, pad_h), (0, pad_w), (0, 0)),
                       mode='constant',
                       constant_values=pad_value)
    
    return padded


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: optimizer
        epoch: current epoch
        metrics: dict of metrics
        filepath: path to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")


def load_checkpoint(model, filepath, device='cpu', optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        filepath: path to checkpoint
        device: device to load to
        optimizer: optional optimizer to load state
    
    Returns:
        epoch: epoch number
        metrics: saved metrics
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"✅ Checkpoint loaded: {filepath} (epoch {epoch})")
    
    return epoch, metrics


def normalize_image(image, mean=None, std=None):
    """
    Normalize image to [-1, 1] or with ImageNet stats.
    
    Args:
        image: (H, W, 3) numpy array in [0, 255]
        mean: normalization mean (default ImageNet)
        std: normalization std (default ImageNet)
    
    Returns:
        normalized: (H, W, 3) float array
    """
    if mean is None:
        mean = np.array(config.IMG_MEAN)
    if std is None:
        std = np.array(config.IMG_STD)
    
    # Convert to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Normalize
    normalized = (image - mean) / std
    
    return normalized


def denormalize_image(image, mean=None, std=None):
    """
    Denormalize image back to [0, 255].
    
    Args:
        image: (H, W, 3) normalized float array
        mean: normalization mean
        std: normalization std
    
    Returns:
        denormalized: (H, W, 3) uint8 array in [0, 255]
    """
    if mean is None:
        mean = np.array(config.IMG_MEAN)
    if std is None:
        std = np.array(config.IMG_STD)
    
    # Denormalize
    image = image * std + mean
    
    # Clip and convert
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    return image


def morphological_cleanup(mask, kernel_size=5, iterations=1):
    """
    Clean up binary mask using morphological operations.
    
    Args:
        mask: (H, W) binary mask
        kernel_size: kernel size for morphology
        iterations: number of iterations
    
    Returns:
        cleaned: cleaned mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                      (kernel_size, kernel_size))
    
    # Opening (remove small noise)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 
                               iterations=iterations)
    
    # Closing (fill small holes)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel,
                               iterations=iterations)
    
    return cleaned


def remove_small_components(mask, min_size=50):
    """
    Remove small connected components.
    
    Args:
        mask: (H, W) binary mask
        min_size: minimum component size in pixels
    
    Returns:
        filtered: filtered mask
    """
    from scipy.ndimage import label, find_objects
    
    # Find connected components
    labeled, num = label(mask > 127)
    
    # Filter small components
    filtered = np.zeros_like(mask)
    
    for i in range(1, num + 1):
        component_mask = (labeled == i)
        if component_mask.sum() >= min_size:
            filtered[component_mask] = 255
    
    return filtered


def edge_detection(image, method='canny', **kwargs):
    """
    Detect edges in image.
    
    Args:
        image: (H, W) or (H, W, 3) image
        method: 'canny', 'sobel', 'laplacian'
        **kwargs: parameters for edge detection
    
    Returns:
        edges: (H, W) edge map
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    if method == 'canny':
        low = kwargs.get('low_threshold', 50)
        high = kwargs.get('high_threshold', 150)
        edges = cv2.Canny(gray, low, high)
    
    elif method == 'sobel':
        ksize = kwargs.get('ksize', 3)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = (edges / edges.max() * 255).astype(np.uint8)
    
    elif method == 'laplacian':
        ksize = kwargs.get('ksize', 3)
        edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        edges = np.abs(edges)
        edges = (edges / edges.max() * 255).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown edge detection method: {method}")
    
    return edges


def create_summary_report(results, save_path=None):
    """
    Create summary report dari evaluation results.
    
    Args:
        results: list of dicts dengan metrics per image
        save_path: path to save report
    
    Returns:
        summary: dict dengan average metrics
    """
    if not results:
        return {}
    
    # Compute averages
    metrics_keys = results[0].keys()
    summary = {}
    
    for key in metrics_keys:
        if isinstance(results[0][key], (int, float)):
            values = [r[key] for r in results]
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)
            summary[f'{key}_min'] = np.min(values)
            summary[f'{key}_max'] = np.max(values)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for key in sorted(summary.keys()):
        if 'mean' in key:
            metric_name = key.replace('_mean', '')
            mean = summary[key]
            std = summary.get(f'{metric_name}_std', 0)
            print(f"{metric_name:12s}: {mean:.4f} ± {std:.4f}")
    
    print("="*60)
    
    # Save to file if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write("EVALUATION SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            for key in sorted(summary.keys()):
                if 'mean' in key:
                    metric_name = key.replace('_mean', '')
                    mean = summary[key]
                    std = summary.get(f'{metric_name}_std', 0)
                    f.write(f"{metric_name:12s}: {mean:.4f} ± {std:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"\n💾 Report saved: {save_path}")
    
    return summary


if __name__ == "__main__":
    # Test utilities
    print("Testing utils...")
    
    # Create dummy data
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    pred = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
    gt = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255
    
    # Test metrics
    metrics = compute_all_metrics(pred, gt)
    print("\nMetrics:", metrics)
    
    # Test overlay
    overlay = overlay_mask_on_image(image, pred)
    print(f"\nOverlay shape: {overlay.shape}")
    
    print("\n✅ Utils test passed!")
