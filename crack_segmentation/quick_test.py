"""
quick_test.py - Quick test script untuk verify setup

Checks:
1. All dependencies installed
2. Dataset accessible
3. Models can be instantiated
4. Basic forward pass works
"""

import sys
import os


def test_imports():
    """Test all required imports"""
    print("\n🔍 Testing imports...")
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"   ❌ PyTorch: {e}")
        return False
    
    try:
        import torchvision
        print(f"   ✅ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"   ❌ TorchVision: {e}")
        return False
    
    try:
        import cv2
        print(f"   ✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"   ❌ OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"   ✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"   ❌ NumPy: {e}")
        return False
    
    try:
        import scipy
        print(f"   ✅ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"   ❌ SciPy: {e}")
        return False
    
    try:
        import matplotlib
        print(f"   ✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"   ❌ Matplotlib: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print(f"   ✅ tqdm")
    except ImportError as e:
        print(f"   ❌ tqdm: {e}")
        return False
    
    return True


def test_modules():
    """Test project modules can be imported"""
    print("\n🔍 Testing project modules...")
    
    try:
        import config
        print(f"   ✅ config.py")
    except ImportError as e:
        print(f"   ❌ config.py: {e}")
        return False
    
    try:
        import dataset
        print(f"   ✅ dataset.py")
    except ImportError as e:
        print(f"   ❌ dataset.py: {e}")
        return False
    
    try:
        import resnet50
        print(f"   ✅ resnet50.py")
    except ImportError as e:
        print(f"   ❌ resnet50.py: {e}")
        return False
    
    try:
        import resnet50_cam
        print(f"   ✅ resnet50_cam.py")
    except ImportError as e:
        print(f"   ❌ resnet50_cam.py: {e}")
        return False
    
    try:
        import resnet50_irn
        print(f"   ✅ resnet50_irn.py")
    except ImportError as e:
        print(f"   ❌ resnet50_irn.py: {e}")
        return False
    
    try:
        import path_index
        print(f"   ✅ path_index.py")
    except ImportError as e:
        print(f"   ❌ path_index.py: {e}")
        return False
    
    try:
        import utils
        print(f"   ✅ utils.py")
    except ImportError as e:
        print(f"   ❌ utils.py: {e}")
        return False
    
    return True


def test_dataset():
    """Test dataset can be accessed"""
    print("\n🔍 Testing dataset...")
    
    import config
    
    # Check directories exist
    if not os.path.exists(config.IMG_DIR):
        print(f"   ❌ Image directory not found: {config.IMG_DIR}")
        return False
    else:
        img_files = [f for f in os.listdir(config.IMG_DIR)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"   ✅ Image directory: {len(img_files)} files")
    
    if not os.path.exists(config.MASK_DIR):
        print(f"   ❌ Mask directory not found: {config.MASK_DIR}")
        return False
    else:
        mask_files = [f for f in os.listdir(config.MASK_DIR)
                      if f.lower().endswith('.png')]
        print(f"   ✅ Mask directory: {len(mask_files)} files")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print(f"   ✅ Output directory: {config.OUTPUT_DIR}")
    
    return True


def test_models():
    """Test models can be instantiated"""
    print("\n🔍 Testing models...")
    
    import torch
    import config
    from resnet50_cam import Net as CamNet, CAM
    from resnet50_irn import AffinityDisplacementLoss
    from path_index import PathIndex
    
    device = config.DEVICE
    print(f"   Device: {device}")
    
    # Test CAM network
    try:
        cam_net = CamNet().to(device)
        print(f"   ✅ CAM Network instantiated")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 256, 256).to(device)
        with torch.no_grad():
            output = cam_net(dummy_input)
        print(f"   ✅ CAM forward pass: input {dummy_input.shape} -> output {output.shape}")
        
        del cam_net, dummy_input, output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"   ❌ CAM Network: {e}")
        return False
    
    # Test CAM extractor
    try:
        cam_extractor = CAM().to(device)
        print(f"   ✅ CAM Extractor instantiated")
        
        # Test forward
        dummy_input = torch.randn(2, 3, 256, 256).to(device)
        with torch.no_grad():
            cam_output = cam_extractor(dummy_input)
        print(f"   ✅ CAM extraction: input {dummy_input.shape} -> output {cam_output.shape}")
        
        del cam_extractor, dummy_input, cam_output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"   ❌ CAM Extractor: {e}")
        return False
    
    # Test IRNet
    try:
        path_idx = PathIndex(feat_h=64, feat_w=64, radius=5)
        print(f"   ✅ PathIndex created")
        
        irnet = AffinityDisplacementLoss(path_idx).to(device)
        print(f"   ✅ IRNet instantiated")
        
        # Test forward (no loss)
        dummy_input = torch.randn(2, 3, 256, 256).to(device)
        with torch.no_grad():
            edge_out, dp_out = irnet(dummy_input, False)
        print(f"   ✅ IRNet forward: edge {edge_out.shape}, displacement {dp_out.shape}")
        
        del path_idx, irnet, dummy_input, edge_out, dp_out
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"   ❌ IRNet: {e}")
        return False
    
    return True


def test_cuda():
    """Test CUDA availability"""
    print("\n🔍 Testing CUDA...")
    
    import torch
    
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test simple operation
        try:
            x = torch.randn(100, 100).cuda()
            y = x @ x.t()
            print(f"   ✅ CUDA tensor operations work")
            del x, y
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ⚠️  CUDA operations failed: {e}")
            return False
    else:
        print(f"   ⚠️  CUDA not available (will use CPU)")
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("QUICK SETUP TEST")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Modules", test_modules()))
    results.append(("Dataset", test_dataset()))
    results.append(("Models", test_models()))
    results.append(("CUDA", test_cuda()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:15s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to go!")
        print("\nNext steps:")
        print("1. Verify your dataset in data/images and data/masks")
        print("2. Adjust config.py if needed")
        print("3. Run: python main.py --stage all")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
