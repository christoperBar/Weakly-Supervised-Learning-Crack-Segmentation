"""
stage5_visualize.py - Visualize Stage 5 training and test segmentation results

Outputs in outputs/stage5_unet:
- training_curves.png
- test_predictions/*.png
- test_visualizations/*_vis.png
- test_metrics_per_image.csv (if GT masks available)
- test_metrics_summary.json (if GT masks available)
- test_metrics_bar.png (if GT masks available)
"""

import csv
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import config
from dataset import get_image_splits
from tesunet import ResNet50UNet, neighborhood_fusion


def _list_images(folder):
    folder_path = Path(folder)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not folder_path.exists():
        return []
    return sorted([p for p in folder_path.iterdir() if p.suffix.lower() in exts])


def _find_mask(mask_dir, image_path):
    if mask_dir is None:
        return None
    mask_root = Path(mask_dir)
    if not mask_root.exists():
        return None

    png_candidate = mask_root / f"{image_path.stem}.png"
    if png_candidate.exists():
        return png_candidate

    same_candidate = mask_root / image_path.name
    if same_candidate.exists():
        return same_candidate

    return None


def _compute_metrics(pred_mask, gt_mask):
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)

    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()

    iou = intersection / (union + 1e-8)
    tp = intersection
    fp = (pred_bin & (1 - gt_bin)).sum()
    fn = ((1 - pred_bin) & gt_bin).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }


def _plot_training_curves(history_path, out_path):
    if not os.path.exists(history_path):
        return False

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    if len(epochs) == 0:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history.get("train_loss", []), label="Train Loss", color="#1f77b4")
    axes[0].plot(epochs, history.get("val_loss", []), label="Val Loss", color="#ff7f0e")
    axes[0].set_title("Stage 5 Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history.get("val_miou", []), label="Val MIoU", color="#2ca02c")
    axes[1].plot(epochs, history.get("val_f1", []), label="Val F1", color="#d62728")
    axes[1].set_title("Stage 5 Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _save_test_metric_reports(metrics_rows, output_dir):
    metric_names = ["IoU", "Precision", "Recall", "F1"]

    csv_path = os.path.join(output_dir, "test_metrics_per_image.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + metric_names)
        for row in metrics_rows:
            writer.writerow([row["filename"]] + [f"{row[m]:.6f}" for m in metric_names])

    values = np.array([[row[m] for m in metric_names] for row in metrics_rows], dtype=np.float32)
    means = values.mean(axis=0)
    stds = values.std(axis=0)

    summary = {
        "count": int(len(metrics_rows)),
        "means": {name: float(means[i]) for i, name in enumerate(metric_names)},
        "stds": {name: float(stds[i]) for i, name in enumerate(metric_names)},
    }

    summary_path = os.path.join(output_dir, "test_metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    x = np.arange(len(metric_names))
    ax.bar(x, means, yerr=stds, capsize=5, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Stage 5 Test Metrics (Mean ± Std)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(means):
        ax.text(i, min(v + 0.02, 0.98), f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    bar_path = os.path.join(output_dir, "test_metrics_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return csv_path, summary_path, bar_path, summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = config.STAGE5_OUTPUT_DIR
    ckpt_path = os.path.join(out_root, "best_model.pth")
    history_path = os.path.join(out_root, "history.json")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    os.makedirs(out_root, exist_ok=True)

    curve_path = os.path.join(out_root, "training_curves_stage5.png")
    curve_ok = _plot_training_curves(history_path, curve_path)

    test_pred_dir = os.path.join(out_root, "test_predictions")
    test_vis_dir = os.path.join(out_root, "test_visualizations")
    os.makedirs(test_pred_dir, exist_ok=True)
    os.makedirs(test_vis_dir, exist_ok=True)

    # Gunakan test split yang sama dengan Stage 1/2
    _, _, test_files = get_image_splits(
        img_dir=config.IMG_DIR,
        n_train=config.N_TRAIN_IMAGES,
        n_val=config.N_VAL_IMAGES,
        n_test=config.N_TEST_IMAGES,
        seed=config.RANDOM_SEED,
    )
    test_images = [Path(config.IMG_DIR) / f for f in test_files]

    if not test_images:
        raise RuntimeError(
            f"No test images found from split in {config.IMG_DIR} with N_TEST_IMAGES={config.N_TEST_IMAGES}"
        )

    model = ResNet50UNet(num_classes=2, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metrics_rows = []
    gt_count = 0

    for img_path in tqdm(test_images, desc="Stage 5 test inference"):
        image_np = np.array(Image.open(img_path).convert("RGB"))
        pred_mask, prob_map = neighborhood_fusion(
            model,
            image_np,
            device,
            patch_size=config.STAGE5_PATCH_SIZE,
            step=80,
            batch_size=max(1, config.STAGE5_BATCH_SIZE // 2),
        )

        pred_u8 = (pred_mask * 255).astype(np.uint8)
        pred_name = f"{img_path.stem}.png"
        pred_path = os.path.join(test_pred_dir, pred_name)
        Image.fromarray(pred_u8).save(pred_path)

        # GT untuk test: gunakan MASK_DIR (ground truth asli)
        gt_path = _find_mask(config.MASK_DIR, img_path)
        gt_mask = None
        if gt_path is not None:
            gt_mask = np.array(Image.open(gt_path).convert("L"))
            if gt_mask.shape != pred_u8.shape:
                gt_mask = cv2.resize(gt_mask, (pred_u8.shape[1], pred_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = _compute_metrics(pred_mask, gt_mask)
            m["filename"] = img_path.name
            metrics_rows.append(m)
            gt_count += 1

        prob_vis = (np.clip(prob_map, 0.0, 1.0) * 255).astype(np.uint8)
        prob_vis = cv2.applyColorMap(prob_vis, cv2.COLORMAP_TURBO)
        prob_vis = cv2.cvtColor(prob_vis, cv2.COLOR_BGR2RGB)

        overlay = image_np.copy()
        crack = pred_mask > 0
        overlay[crack] = (0.55 * overlay[crack] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)

        # Tambahkan panel khusus untuk mask biner selain overlay
        cols = 5 if gt_mask is not None else 4
        fig, axes = plt.subplots(1, cols, figsize=(4.5 * cols, 4.5))
        if cols == 4:
            axes = list(axes)

        axes[0].imshow(image_np)
        axes[0].set_title("Test Image")
        axes[0].axis("off")

        axes[1].imshow(prob_vis)
        axes[1].set_title("Crack Probability")
        axes[1].axis("off")

        axes[2].imshow(pred_u8, cmap="gray")
        axes[2].set_title("Binary Prediction")
        axes[2].axis("off")

        axes[3].imshow(overlay)
        axes[3].set_title("Prediction Overlay")
        axes[3].axis("off")

        if gt_mask is not None:
            axes[4].imshow(gt_mask, cmap="gray")
            axes[4].set_title("Ground Truth")
            axes[4].axis("off")

        plt.tight_layout()
        vis_path = os.path.join(test_vis_dir, f"{img_path.stem}_vis.png")
        plt.savefig(vis_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    metrics_summary = None
    if metrics_rows:
        _, _, _, metrics_summary = _save_test_metric_reports(metrics_rows, out_root)

    print("\n" + "=" * 60)
    print("STAGE 5 VISUALIZATION COMPLETED")
    print("=" * 60)
    if curve_ok:
        print(f"Training curves : {curve_path}")
    else:
        print("Training curves : history.json not found or empty")
    print(f"Test masks      : {test_pred_dir}")
    print(f"Test visuals    : {test_vis_dir}")
    print(f"Test images     : {len(test_images)}")
    print(f"GT available    : {gt_count}")

    if metrics_summary is not None:
        print("\nMean test metrics:")
        for k, v in metrics_summary["means"].items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
