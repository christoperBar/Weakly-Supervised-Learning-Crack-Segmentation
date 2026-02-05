"""
train_stage1.py - Stage 1: Train Classification Network (CAM)

Melatih ResNet50-based classifier untuk crack detection.
Output: CAM network yang dapat menghasilkan Class Activation Maps.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import modules
import config
from dataset import CrackPatchDataset, get_train_transform
from resnet50_cam import Net as CamNet, NUM_CLASSES


class FocalLoss(nn.Module):
    """
    Focal Loss untuk handling class imbalance.
    Fokus lebih pada hard examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, num_classes) raw logits
            targets: (B,) class indices
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train satu epoch"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    # Confusion matrix
    tp, fp, tn, fn = 0, 0, 0, 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            # Confusion matrix for crack class
            for p, l in zip(pred, labels):
                if l == 1:  # Ground truth = crack
                    if p == 1:
                        tp += 1
                    else:
                        fn += 1
                else:  # Ground truth = background
                    if p == 1:
                        fp += 1
                    else:
                        tn += 1
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    # Precision, Recall, F1 untuk crack class
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    print(f"\n📊 Evaluation Results:")
    print(f"   Loss: {avg_loss:.4f}")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    return avg_loss, accuracy, precision, recall, f1


def plot_training_history(history, save_path='outputs/stage1_training.png'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Training plot saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("STAGE 1: Training Classification Network (CAM)")
    print("=" * 60)
    
    # Device
    device = config.DEVICE
    print(f"\n🖥️  Device: {device}")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Dataset
    print(f"\n📦 Loading dataset...")
    train_transform = get_train_transform()
    train_dataset = CrackPatchDataset(
        img_dir=config.IMG_DIR,
        mask_dir=config.MASK_DIR,
        patch_size=config.PATCH_SIZE,
        stride=config.PATCH_STRIDE,
        transform=train_transform,
        max_neg_ratio=config.MAX_NEG_RATIO,
        min_crack_ratio=config.MIN_CRACK_RATIO
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.CAM_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"   Batch size: {config.CAM_BATCH_SIZE}")
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Model
    print(f"\n🏗️  Building model...")
    model = CamNet().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss & Optimizer
    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.CAM_LR,
        weight_decay=config.CAM_WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.CAM_EPOCHS,
        eta_min=1e-6
    )
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {config.CAM_EPOCHS}")
    print(f"   Learning rate: {config.CAM_LR}")
    
    history = {
        'train_loss': [],
        'train_acc': []
    }
    
    best_acc = 0
    best_epoch = 0
    
    for epoch in range(config.CAM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.CAM_EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Train Acc: {train_acc:.2f}%")
        print(f"   Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if train_acc > best_acc:
            best_acc = train_acc
            best_epoch = epoch + 1
            save_path = os.path.join(config.OUTPUT_DIR, 'cam_net_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'train_loss': train_loss,
            }, save_path)
            print(f"   💾 Best model saved! (Acc: {best_acc:.2f}%)")
    
    # Save final model
    final_path = os.path.join(config.OUTPUT_DIR, 'cam_net_final.pth')
    torch.save({
        'epoch': config.CAM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'train_loss': train_loss,
    }, final_path)
    print(f"\n💾 Final model saved: {final_path}")
    
    # Plot training history
    plot_training_history(history)
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    print(f"Final Accuracy: {train_acc:.2f}%")
    print(f"Models saved in: {config.OUTPUT_DIR}")
    
    return model


if __name__ == "__main__":
    model = main()
