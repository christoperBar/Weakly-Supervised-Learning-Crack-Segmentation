"""
example_usage.py - Contoh penggunaan kode untuk berbagai use cases

Examples:
1. Data preparation
2. Training from scratch
3. Resume training
4. Inference on new images
5. Evaluation on test set
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path

import config
from dataset import CrackPatchDataset, CrackFullImageDataset, get_train_transform, get_inference_transform
from resnet50_cam import Net as CamNet, CAM
from resnet50_irn import AffinityDisplacementLoss, EdgeDisplacement
from path_index import PathIndex
from utils import compute_all_metrics, visualize_comparison, create_summary_report


# ============================================================
# Example 1: Data Preparation
# ============================================================
def example_data_preparation():
    """Contoh mempersiapkan dataset"""
    print("\n" + "="*60)
    print("Example 1: Data Preparation")
    print("="*60)
    
    # Verifikasi dataset
    from dataset import verify_dataset
    matched, unmatched = verify_dataset(config.IMG_DIR, config.MASK_DIR)
    
    print(f"\n✅ Dataset verification complete:")
    print(f"   Matched: {matched}")
    print(f"   Unmatched: {len(unmatched)}")
    
    # Create patch dataset
    train_transform = get_train_transform()
    dataset = CrackPatchDataset(
        img_dir=config.IMG_DIR,
        mask_dir=config.MASK_DIR,
        patch_size=config.PATCH_SIZE,
        stride=config.PATCH_STRIDE,
        transform=train_transform
    )
    
    print(f"\n✅ Patch dataset created:")
    print(f"   Total patches: {len(dataset)}")
    
    # Sample a patch
    if len(dataset) > 0:
        patch, label = dataset[0]
        print(f"\n📊 Sample patch:")
        print(f"   Shape: {patch.shape}")
        print(f"   Label: {'Crack' if label == 1 else 'Background'}")


# ============================================================
# Example 2: Training from Scratch
# ============================================================
def example_training_from_scratch():
    """Contoh training dari awal"""
    print("\n" + "="*60)
    print("Example 2: Training from Scratch")
    print("="*60)
    
    print("\nTo train from scratch, run:")
    print("  python main.py --stage all")
    print("\nOr step by step:")
    print("  python main.py --stage 1   # Train CAM network")
    print("  python main.py --stage 2   # Train IRNet")
    print("  python main.py --stage 3   # Generate pseudo labels")


# ============================================================
# Example 3: Resume Training
# ============================================================
def example_resume_training():
    """Contoh melanjutkan training"""
    print("\n" + "="*60)
    print("Example 3: Resume Training")
    print("="*60)
    
    device = config.DEVICE
    
    # Load checkpoint
    checkpoint_path = os.path.join(config.OUTPUT_DIR, 'cam_net_best.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"\n⚠️  No checkpoint found at {checkpoint_path}")
        print("   Train a model first using: python main.py --stage 1")
        return
    
    # Create model
    model = CamNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.CAM_LR)
    
    # Load checkpoint
    from utils import load_checkpoint
    epoch, metrics = load_checkpoint(model, checkpoint_path, device, optimizer)
    
    print(f"\n✅ Checkpoint loaded:")
    print(f"   Epoch: {epoch}")
    print(f"   Metrics: {metrics}")
    
    print("\n💡 You can now continue training from epoch", epoch)


# ============================================================
# Example 4: Inference on New Images
# ============================================================
def example_inference_single_image():
    """Contoh inference pada single image"""
    print("\n" + "="*60)
    print("Example 4: Inference on Single Image")
    print("="*60)
    
    device = config.DEVICE
    
    # Check if models exist
    cam_path = os.path.join(config.OUTPUT_DIR, 'cam_net_best.pth')
    irnet_path = os.path.join(config.OUTPUT_DIR, 'irnet_best.pth')
    
    if not os.path.exists(cam_path) or not os.path.exists(irnet_path):
        print("\n⚠️  Models not found. Train first using:")
        print("   python main.py --stage all")
        return
    
    # Load models
    print("\n📥 Loading models...")
    cam_net = CAM().to(device)
    checkpoint = torch.load(cam_path, map_location=device)
    cam_net.load_state_dict(checkpoint['model_state_dict'])
    cam_net.eval()
    
    irnet = EdgeDisplacement(crop_size=512).to(device)
    checkpoint = torch.load(irnet_path, map_location=device)
    irnet.load_state_dict(checkpoint['model_state_dict'])
    irnet.eval()
    
    print("   ✅ Models loaded")
    
    # Get first image from dataset
    img_files = [f for f in os.listdir(config.IMG_DIR)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not img_files:
        print("\n⚠️  No images found in dataset")
        return
    
    img_path = os.path.join(config.IMG_DIR, img_files[0])
    print(f"\n🎨 Processing: {img_files[0]}")
    
    # Load image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"   Image size: {img_rgb.shape}")
    print(f"\n💡 For full inference pipeline, run:")
    print(f"   python main.py --inference")


# ============================================================
# Example 5: Batch Evaluation
# ============================================================
def example_batch_evaluation():
    """Contoh evaluasi pada test set"""
    print("\n" + "="*60)
    print("Example 5: Batch Evaluation")
    print("="*60)
    
    # Check if pseudo labels exist
    pseudo_dir = os.path.join(config.OUTPUT_DIR, 'pseudo_labels')
    
    if not os.path.exists(pseudo_dir):
        print("\n⚠️  Pseudo labels not found. Generate first using:")
        print("   python main.py --stage 3")
        return
    
    # Get all predictions and ground truths
    pseudo_files = sorted([f for f in os.listdir(pseudo_dir)
                          if f.endswith('_pseudo.png')])
    
    if not pseudo_files:
        print("\n⚠️  No pseudo labels found")
        return
    
    print(f"\n📊 Evaluating {len(pseudo_files)} predictions...")
    
    results = []
    
    for fname in pseudo_files[:5]:  # Evaluate first 5 for example
        # Load prediction
        pred_path = os.path.join(pseudo_dir, fname)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        # Load ground truth
        gt_name = fname.replace('_pseudo.png', '.png')
        gt_path = os.path.join(config.MASK_DIR, gt_name)
        
        if not os.path.exists(gt_path):
            continue
        
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Compute metrics
        metrics = compute_all_metrics(pred, gt)
        metrics['filename'] = fname
        results.append(metrics)
        
        print(f"   {fname}: IoU={metrics['IoU']:.4f}, F1={metrics['F1']:.4f}")
    
    # Summary
    if results:
        summary = create_summary_report(results)


# ============================================================
# Example 6: Custom Data Augmentation
# ============================================================
def example_custom_augmentation():
    """Contoh custom augmentation"""
    print("\n" + "="*60)
    print("Example 6: Custom Data Augmentation")
    print("="*60)
    
    from torchvision import transforms
    
    # Define custom transform
    custom_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),  # Increased rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translation
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)
    ])
    
    print("\n✅ Custom augmentation defined")
    print("   Modifications:")
    print("   - Increased rotation to ±30°")
    print("   - Added saturation jitter")
    print("   - Added random translation")
    
    # Create dataset with custom transform
    dataset = CrackPatchDataset(
        img_dir=config.IMG_DIR,
        mask_dir=config.MASK_DIR,
        patch_size=config.PATCH_SIZE,
        transform=custom_transform
    )
    
    print(f"\n✅ Dataset with custom augmentation created")


# ============================================================
# Example 7: Visualize Model Predictions
# ============================================================
def example_visualize_predictions():
    """Contoh visualisasi predictions"""
    print("\n" + "="*60)
    print("Example 7: Visualize Predictions")
    print("="*60)
    
    vis_dir = os.path.join(config.OUTPUT_DIR, 'visualizations')
    
    if not os.path.exists(vis_dir):
        print("\n⚠️  Visualizations not found. Generate using:")
        print("   python main.py --stage 3")
        return
    
    vis_files = [f for f in os.listdir(vis_dir) if f.endswith('_vis.png')]
    
    if vis_files:
        print(f"\n✅ Found {len(vis_files)} visualization files")
        print(f"   Location: {vis_dir}")
        print(f"   Example: {vis_files[0]}")
    else:
        print("\n⚠️  No visualization files found")


# ============================================================
# Main
# ============================================================
def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("CRACK SEGMENTATION - USAGE EXAMPLES")
    print("="*60)
    
    examples = [
        ("Data Preparation", example_data_preparation),
        ("Training from Scratch", example_training_from_scratch),
        ("Resume Training", example_resume_training),
        ("Inference on New Images", example_inference_single_image),
        ("Batch Evaluation", example_batch_evaluation),
        ("Custom Augmentation", example_custom_augmentation),
        ("Visualize Predictions", example_visualize_predictions),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n⚠️  Error in {name}: {e}")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
