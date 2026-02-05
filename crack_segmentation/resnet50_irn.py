"""
resnet50_irn.py  —  IRNet: two-branch network (edge + displacement).

Faithful to the official IRNet repo.  No structural changes whatsoever.
Comments added to map every layer back to the paper (Sec 4).

Classes
-------
Net                       — base two-branch model
AffinityDisplacementLoss  — wraps Net; adds vectorised loss computation
EdgeDisplacement          — wraps Net; inference-time pad/crop + TTA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet50


class Net(nn.Module):
    """
    IRNet base model.

    Backbone stages (strides=[2,2,2,1]):
        stage1: conv1+bn1+relu+maxpool        →  64 ch,  /4
        stage2: layer1                        → 256 ch,  /4
        stage3: layer2                        → 512 ch,  /8
        stage4: layer3                        →1024 ch, /16
        stage5: layer4  (stride=1)            →2048 ch, /16

    Edge (boundary) branch  —  Sec 4, boundary-detection branch:
        each stage → 32 ch (with upsample to /4 where needed)
        concat 5×32 = 160 ch → 1×1 conv → 1 ch  (class boundary map)

    Displacement branch  —  Sec 4, displacement-field branch:
        3-level top-down merge:
            dp3,dp4,dp5 concat at /8 resolution → dp6 (up to /4)
            dp1,dp2,dp6 concat at /4 resolution → dp7  → 2 ch output
        MeanShift applied at the very end (identity during training).
    """

    def __init__(self):
        super(Net, self).__init__()

        # ── backbone ──────────────────────────────────────────────────
        self.resnet50 = resnet50.resnet50(pretrained=True, strides=[2, 2, 2, 1])

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1,
                                    self.resnet50.relu, self.resnet50.maxpool)
        self.stage2 = nn.Sequential(self.resnet50.layer1)
        self.stage3 = nn.Sequential(self.resnet50.layer2)
        self.stage4 = nn.Sequential(self.resnet50.layer3)
        self.stage5 = nn.Sequential(self.resnet50.layer4)
        self.mean_shift = Net.MeanShift(2)

        # ── edge (boundary) branch ────────────────────────────────────
        # stage1  64 ch  /4   → no upsample needed
        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        # stage2 256 ch  /4   → no upsample needed
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        # stage3 512 ch  /8   → 2× upsample → /4
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        # stage4 1024ch /16   → 4× upsample → /4
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        # stage5 2048ch /16   → 4× upsample → /4
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        # final 1×1 conv: 5×32 = 160 → 1  (no GN / ReLU — raw logit)
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)

        # ── displacement branch ───────────────────────────────────────
        # stage1  64 ch  /4
        self.fc_dp1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        # stage2 256 ch  /4
        self.fc_dp2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
        )
        # stage3 512 ch  /8
        self.fc_dp3 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
        )
        # stage4 1024ch /16  → 2× up → /8
        self.fc_dp4 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        # stage5 2048ch /16  → 2× up → /8
        self.fc_dp5 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        # merge at /8: concat(dp3, dp4, dp5) = 256+256+256 = 768 → 256, then 2× up → /4
        self.fc_dp6 = nn.Sequential(
            nn.Conv2d(768, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        # final merge at /4: concat(dp1, dp2, dp_up3) = 64+128+256 = 448 → 256 → 2
        self.fc_dp7 = nn.Sequential(
            nn.Conv2d(448, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1, bias=False),
            self.mean_shift                          # identity in train, subtract mean in eval
        )

        # ── module lists (used by .trainable_parameters()) ────────────
        self.backbone    = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.edge_layers = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge3,
                                          self.fc_edge4, self.fc_edge5, self.fc_edge6])
        self.dp_layers   = nn.ModuleList([self.fc_dp1, self.fc_dp2, self.fc_dp3,
                                          self.fc_dp4, self.fc_dp5, self.fc_dp6, self.fc_dp7])

    # ── MeanShift: corrects displacement-field bias at inference ──────
    class MeanShift(nn.Module):
        def __init__(self, num_features):
            super(Net.MeanShift, self).__init__()
            self.register_buffer('running_mean', torch.zeros(num_features))

        def forward(self, input):
            if self.training:
                return input
            return input - self.running_mean.view(1, 2, 1, 1)

    # ── forward ───────────────────────────────────────────────────────
    def forward(self, x):
        # backbone  — .detach() freezes gradients per stage
        x1 = self.stage1(x).detach()     #  64 ch  /4
        x2 = self.stage2(x1).detach()    # 256 ch  /4
        x3 = self.stage3(x2).detach()    # 512 ch  /8
        x4 = self.stage4(x3).detach()    #1024 ch /16
        x5 = self.stage5(x4).detach()    #2048 ch /16

        # ── edge branch ─────────────────────────────────────────────
        edge1 = self.fc_edge1(x1)                                          # /4
        edge2 = self.fc_edge2(x2)                                          # /4  ← reference size
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]     # crop to edge2 size
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]
        edge_out = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))
        # edge_out: (B, 1, H/4, W/4) — raw logit; sigmoid applied later

        # ── displacement branch ───────────────────────────────────
        dp1 = self.fc_dp1(x1)                                              # 64  ch  /4
        dp2 = self.fc_dp2(x2)                                              # 128 ch  /4
        dp3 = self.fc_dp3(x3)                                              # 256 ch  /8  ← ref
        dp4 = self.fc_dp4(x4)[..., :dp3.size(2), :dp3.size(3)]            # 256 ch  /8
        dp5 = self.fc_dp5(x5)[..., :dp3.size(2), :dp3.size(3)]            # 256 ch  /8

        dp_up3 = self.fc_dp6(torch.cat([dp3, dp4, dp5], dim=1))[..., :dp2.size(2), :dp2.size(3)]
        dp_out = self.fc_dp7(torch.cat([dp1, dp2, dp_up3], dim=1))
        # dp_out: (B, 2, H/4, W/4)

        return edge_out, dp_out

    def trainable_parameters(self):
        """Return only branch parameters (backbone is frozen via detach)."""
        return (tuple(self.edge_layers.parameters()),
                tuple(self.dp_layers.parameters()))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()            # keep backbone BN in eval mode


# ══════════════════════════════════════════════════════════════════════
# AffinityDisplacementLoss
# ══════════════════════════════════════════════════════════════════════
class AffinityDisplacementLoss(Net):
    """
    Extends Net with vectorised loss tensors.

    Requires a PathIndex object (see path_index.py) that pre-computes:
        .path_indices   – list of index arrays, one per path-length bucket
        .search_dst     – (num_directions, 2) array of (dy, dx) offsets
        .radius_floor   – floor(radius)

    forward(x, return_loss):
        if return_loss is False → returns (edge_out, dp_out) raw
        if True                 → returns (pos_aff_loss, neg_aff_loss,
                                           dp_fg_loss, dp_bg_loss)
            The CALLER is responsible for masking these with the P+/P- labels
            from the CAM-mined inter-pixel relations.
    """

    path_indices_prefix = "path_indices"

    def __init__(self, path_index):
        super(AffinityDisplacementLoss, self).__init__()

        self.path_index = path_index

        # register each path-index array as a buffer (moves to GPU with .to())
        self.n_path_lengths = len(path_index.path_indices)
        for i, pi in enumerate(path_index.path_indices):
            self.register_buffer(
                AffinityDisplacementLoss.path_indices_prefix + str(i),
                torch.from_numpy(pi)
            )

        # displacement target: (dy, dx) for every search direction
        # shape after unsqueeze: (1, 2, num_directions, 1)
        self.register_buffer(
            'disp_target',
            torch.unsqueeze(
                torch.unsqueeze(
                    torch.from_numpy(path_index.search_dst).transpose(1, 0), 0),
                -1
            ).float()
        )

    # ── Eq (7):  a_ij = 1 − max_{k ∈ Π_ij} B(x_k) ─────────────────
    def to_affinity(self, edge):
        """
        edge : (B, 1, H, W) after sigmoid  →  flatten to (B, H*W)
        Uses pre-indexed paths to gather B values along each pixel-pair line,
        then max-pools over the path length  →  affinity = 1 - max.
        """
        aff_list = []
        edge = edge.view(edge.size(0), -1)                       # (B, N)

        for i in range(self.n_path_lengths):
            ind      = self._buffers[AffinityDisplacementLoss.path_indices_prefix + str(i)]
            ind_flat = ind.view(-1)                               # flatten index array
            dist     = torch.index_select(edge, dim=-1, index=ind_flat)
            dist     = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
            # max over path-length dim  →  Eq (7)
            aff      = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
            aff_list.append(aff)

        aff_cat = torch.cat(aff_list, dim=1)                     # (B, num_pairs)
        return aff_cat

    # ── Eq (5): δ(i,j) = D(x_i) − D(x_j) for every search direction ──
    def to_pair_displacement(self, disp):
        """
        disp : (B, 2, H, W)
        For each (dy, dx) in search_dst, shift the map and subtract.
        Returns pair_disp : (B, 2, num_directions, H', W')  reshaped to (..., H'*W')
        """
        height, width    = disp.size(2), disp.size(3)
        radius_floor     = self.path_index.radius_floor

        cropped_height   = height - radius_floor
        cropped_width    = width  - 2 * radius_floor

        # source crop (centre region)
        disp_src = disp[:, :, :cropped_height, radius_floor:radius_floor + cropped_width]

        # destination crops for each (dy, dx)
        disp_dst = [
            disp[:, :, dy:dy + cropped_height, radius_floor + dx:radius_floor + dx + cropped_width]
            for dy, dx in self.path_index.search_dst
        ]
        disp_dst = torch.stack(disp_dst, 2)                      # (B, 2, D, H', W')

        pair_disp = torch.unsqueeze(disp_src, 2) - disp_dst      # (B, 2, D, H', W')
        pair_disp = pair_disp.view(pair_disp.size(0), pair_disp.size(1),
                                   pair_disp.size(2), -1)         # (B, 2, D, H'*W')
        return pair_disp

    # ── Eq (5): |δ(i,j) − δ̂(i,j)|  ────────────────────────────────
    def to_displacement_loss(self, pair_disp):
        """L1 between predicted pair-displacement and ground-truth offset."""
        return torch.abs(pair_disp - self.disp_target)

    def forward(self, *inputs):
        x, return_loss = inputs
        edge_out, dp_out = super().forward(x)

        if return_loss is False:
            return edge_out, dp_out

        # ── affinity losses  (Eq 8) ─────────────────────────────────
        aff            = self.to_affinity(torch.sigmoid(edge_out))
        pos_aff_loss   = (-1) * torch.log(aff + 1e-5)            # −log(a)
        neg_aff_loss   = (-1) * torch.log(1. + 1e-5 - aff)       # −log(1−a)

        # ── displacement losses  (Eq 5 & 6) ─────────────────────────
        pair_disp      = self.to_pair_displacement(dp_out)
        dp_fg_loss     = self.to_displacement_loss(pair_disp)     # |δ − δ̂|
        dp_bg_loss     = torch.abs(pair_disp)                     # |δ|

        # raw tensors returned; caller applies P+_fg / P+_bg / P- masks
        return pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss


# ══════════════════════════════════════════════════════════════════════
# EdgeDisplacement  —  inference wrapper
# ══════════════════════════════════════════════════════════════════════
class EdgeDisplacement(Net):
    """
    Pads the input to crop_size so that the backbone produces a clean
    feature map, runs forward, then crops back.  Also applies TTA
    (average of original and h-flipped edge maps over batch dim).
    """

    def __init__(self, crop_size=512, stride=4):
        super(EdgeDisplacement, self).__init__()
        self.crop_size = crop_size
        self.stride    = stride

    def forward(self, x):
        # expected output spatial size (before padding)
        feat_size = ((x.size(2) - 1) // self.stride + 1,
                     (x.size(3) - 1) // self.stride + 1)

        # pad to crop_size × crop_size
        x = F.pad(x, [0, self.crop_size - x.size(3),
                       0, self.crop_size - x.size(2)])

        edge_out, dp_out = super().forward(x)

        # crop outputs back to expected size
        edge_out = edge_out[..., :feat_size[0], :feat_size[1]]
        dp_out   = dp_out[...,  :feat_size[0], :feat_size[1]]

        # TTA over batch dim: edge_out[0] is original, edge_out[1] is h-flipped
        edge_out = torch.sigmoid(edge_out[0] / 2 + edge_out[1].flip(-1) / 2)
        dp_out   = dp_out[0]                                     # just take first (original)

        return edge_out, dp_out
