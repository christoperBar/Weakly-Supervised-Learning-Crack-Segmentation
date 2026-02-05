"""
main.py - Full pipeline untuk Crack Weakly Supervised Segmentation

Pipeline lengkap:
1. Stage 1: Train classification network (CAM)
2. Stage 2+3: Mine inter-pixel relations dan train IRNet
3. Stage 4: Generate pseudo instance segmentation labels
4. Evaluation (optional)

Usage:
    python main.py --stage all              # Run all stages
    python main.py --stage 1                # Run stage 1 only
    python main.py --stage 2                # Run stage 2+3 only
    python main.py --stage 3                # Run stage 4 (inference) only
    python main.py --inference              # Inference only mode
"""

import argparse
import os
import sys

import config


def run_stage1():
    """Stage 1: Train CAM network"""
    print("\n" + "="*60)
    print("STAGE 1: Training Classification Network")
    print("="*60)
    
    from train_stage1 import main as train_cam
    model = train_cam()
    
    return model


def run_stage2_3():
    """Stage 2+3: Mine relations and train IRNet"""
    print("\n" + "="*60)
    print("STAGE 2+3: Mine Relations & Train IRNet")
    print("="*60)
    
    # Check if CAM network exists
    cam_path = os.path.join(config.OUTPUT_DIR, 'cam_net_best.pth')
    if not os.path.exists(cam_path):
        print(f"\n❌ Error: CAM network not found at {cam_path}")
        print(f"   Please run Stage 1 first!")
        sys.exit(1)
    
    from train_stage2_3 import main as train_irnet
    model = train_irnet()
    
    return model


def run_inference():
    """Stage 4: Generate pseudo labels"""
    print("\n" + "="*60)
    print("STAGE 4: Generating Pseudo Labels")
    print("="*60)
    
    # Check if models exist
    cam_path = os.path.join(config.OUTPUT_DIR, 'cam_net_best.pth')
    irnet_path = os.path.join(config.OUTPUT_DIR, 'irnet_best.pth')
    
    if not os.path.exists(cam_path):
        print(f"\n❌ Error: CAM network not found at {cam_path}")
        print(f"   Please run Stage 1 first!")
        sys.exit(1)
    
    if not os.path.exists(irnet_path):
        print(f"\n❌ Error: IRNet not found at {irnet_path}")
        print(f"   Please run Stage 2+3 first!")
        sys.exit(1)
    
    from inference import main as run_inference_main
    run_inference_main()


def verify_setup():
    """Verify dataset and directory structure"""
    print("\n🔍 Verifying setup...")
    
    # Check directories
    if not os.path.exists(config.IMG_DIR):
        print(f"❌ Error: Image directory not found: {config.IMG_DIR}")
        return False
    
    if not os.path.exists(config.MASK_DIR):
        print(f"❌ Error: Mask directory not found: {config.MASK_DIR}")
        return False
    
    # Check if images exist
    img_files = [f for f in os.listdir(config.IMG_DIR)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(img_files) == 0:
        print(f"❌ Error: No images found in {config.IMG_DIR}")
        return False
    
    print(f"✅ Found {len(img_files)} images in {config.IMG_DIR}")
    
    # Check masks
    mask_files = [f for f in os.listdir(config.MASK_DIR)
                  if f.lower().endswith('.png')]
    
    print(f"✅ Found {len(mask_files)} masks in {config.MASK_DIR}")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"✅ Output directory: {config.OUTPUT_DIR}")
    
    return True


def print_config():
    """Print current configuration"""
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"📁 Data:")
    print(f"   Image directory: {config.IMG_DIR}")
    print(f"   Mask directory: {config.MASK_DIR}")
    print(f"   Output directory: {config.OUTPUT_DIR}")
    print(f"\n📐 Image:")
    print(f"   Resolution: {config.IMG_HEIGHT}x{config.IMG_WIDTH}")
    print(f"   Patch size: {config.PATCH_SIZE}")
    print(f"   Patch stride: {config.PATCH_STRIDE}")
    print(f"\n🎓 Training:")
    print(f"   CAM epochs: {config.CAM_EPOCHS}")
    print(f"   IRN epochs: {config.IRN_EPOCHS}")
    print(f"   CAM batch size: {config.CAM_BATCH_SIZE}")
    print(f"   IRN batch size: {config.IRN_BATCH_SIZE}")
    print(f"   Device: {config.DEVICE}")
    print(f"\n🎯 Pseudo Label:")
    print(f"   Random walk steps: {config.T_WALK}")
    print(f"   Beta: {config.BETA}")
    print(f"   Background quantile: {config.BG_QUANTILE}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Crack Weakly Supervised Segmentation Pipeline'
    )
    parser.add_argument(
        '--stage',
        type=str,
        choices=['all', '1', '2', '3'],
        default='all',
        help='Which stage to run (all/1/2/3)'
    )
    parser.add_argument(
        '--inference',
        action='store_true',
        help='Run inference only (skip training)'
    )
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='Skip dataset verification'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*60)
    print("  CRACK WEAKLY SUPERVISED SEGMENTATION")
    print("  Based on IRNet (Ahn et al., CVPR 2019)")
    print("  Dataset: 4032x3024 crack images")
    print("="*60)
    
    # Print configuration
    print_config()
    
    # Verify setup
    if not args.skip_verify:
        if not verify_setup():
            print("\n❌ Setup verification failed!")
            sys.exit(1)
    
    # Run stages
    if args.inference:
        # Inference only
        run_inference()
    
    elif args.stage == 'all':
        # Run all stages
        print("\n🚀 Running full pipeline (all stages)...")
        run_stage1()
        run_stage2_3()
        run_inference()
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED!")
        print("="*60)
        print(f"Results saved in: {config.OUTPUT_DIR}")
        
    elif args.stage == '1':
        # Stage 1 only
        run_stage1()
        
        print("\n✅ Stage 1 completed!")
        print("   Next: Run --stage 2 to train IRNet")
        
    elif args.stage == '2':
        # Stage 2+3 only
        run_stage2_3()
        
        print("\n✅ Stage 2+3 completed!")
        print("   Next: Run --stage 3 or --inference to generate pseudo labels")
        
    elif args.stage == '3':
        # Stage 4 (inference) only
        run_inference()
        
        print("\n✅ Stage 4 completed!")
        print(f"   Pseudo labels saved in: {config.OUTPUT_DIR}/pseudo_labels")


if __name__ == "__main__":
    main()