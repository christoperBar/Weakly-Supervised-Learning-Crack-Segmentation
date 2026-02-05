"""
resnet50_cam.py  —  Classification network + CAM extraction.

Faithful to the official IRNet repo.  Only change from the original:
    num_classes  20 (PASCAL VOC)  →  2 (background / crack)

The CAM class forward() expects a batch of 2 images:
    [original, horizontal-flip]
and returns:  cam[0] + cam[1].flip(-1)   (TTA built in).
If you feed a single image, use batch_size=1 and call .forward() directly
on the Net class instead, or duplicate+flip yourself.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet50

NUM_CLASSES = 2          # 0 = background, 1 = crack


# ── tiny utility matching torchutils.gap2d from the official repo ──
def gap2d(x, keepdims=False):
    """Global average pooling over spatial dims."""
    x = x.mean(dim=(2, 3), keepdim=keepdims)
    return x


class Net(nn.Module):
    """
    Image-classification network.
    strides=(2,2,2,1) → layer4 stride=1 → feature map is input/16.

    Stage grouping (same as official repo):
        stage1 = conv1 + bn1 + relu + maxpool + layer1   (256 ch, /4)
        stage2 = layer2                                  (512 ch, /8)
        stage3 = layer3                                  (1024 ch, /16)
        stage4 = layer4                                  (2048 ch, /16)
    """

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(
            self.resnet50.conv1, self.resnet50.bn1,
            self.resnet50.relu, self.resnet50.maxpool,
            self.resnet50.layer1
        )
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        # 1×1 conv classifier  (official uses Conv2d, NOT nn.Linear)
        self.classifier = nn.Conv2d(2048, NUM_CLASSES, 1, bias=False)

        self.backbone   = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    # ── classification forward (with GAP) ──
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x).detach()   # detach between stage2 and stage3 (official pattern)
        x = self.stage3(x)
        x = self.stage4(x)

        x = gap2d(x, keepdims=True)   # (B, 2048, 1, 1)
        x = self.classifier(x)        # (B, NUM_CLASSES, 1, 1)
        x = x.view(-1, NUM_CLASSES)
        return x

    def train(self, mode=True):
        super().train(mode)
        # freeze conv1 + bn1 (official pattern)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):
    """
    Extracts raw (un-normalised) CAMs.

    forward() expects batch = 2:  [img, img.flip(-1)]
    Returns:  cam[0] + cam[1].flip(-1)   shape (NUM_CLASSES, H/16, W/16)

    If you only have a single image, just use batch=1 and skip the TTA:
        cam = F.conv2d(features, self.classifier.weight)
        cam = F.relu(cam)[0]
    """

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Eq (1): φ_c^T f(x)  via 1×1 conv  +  ReLU
        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)

        # TTA: average original and h-flipped
        x = x[0] + x[1].flip(-1)     # (NUM_CLASSES, h, w)
        return x
