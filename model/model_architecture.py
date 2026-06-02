# model_architecture.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# CONFIGURATION
# ============================================================
NUM_CLASSES    = 4      # Grade 0, 1, 2, 3
DROPOUT_RATE   = 0.4
EMBED_DIM      = 256    # ConvFormer embedding dimension
NUM_HEADS      = 8      # Attention heads in ConvFormer
DEPTH          = 4      # Number of ConvFormer blocks
MLP_RATIO      = 4      # MLP expansion ratio in ConvFormer
INPUT_SIZE     = 256    # Spatial resolution after augmentation


# ============================================================
# BLOCK 1: CNN BACKBONE
# Extracts local texture features:
# lipid droplet boundaries, nuclear morphology,
# cellular density patterns
# ============================================================
class CNNBackbone(nn.Module):
    """
    EfficientNet-B4 backbone pretrained on ImageNet.
    Final classifier replaced with a feature projection
    layer to produce a fixed-dimension feature vector.

    Local feature extraction rationale:
        EfficientNet-B4 compound scaling provides an optimal
        balance between parameter efficiency and receptive
        field coverage for 256×256 histopathology tiles.
        ImageNet pretraining provides robust low-level
        feature detectors transferable to H&E texture.

    Output:
        Tensor of shape [B, 256] — local feature vector
    """
    def __init__(self, out_dim=EMBED_DIM, pretrained=True):
        super().__init__()

        # Load pretrained EfficientNet-B4
        weights = (
            models.EfficientNet_B4_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        backbone = models.efficientnet_b4(weights=weights)

        # Remove final classifier — keep feature extractor only
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Project to shared embedding dimension
        self.projector = nn.Sequential(
            nn.Linear(1792, out_dim),  # EfficientNet-B4 → 1792 channels
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE)
        )

    def forward(self, x):
        x = self.features(x)          # [B, 1792, H', W']
        x = self.pool(x)              # [B, 1792, 1, 1]
        x = x.flatten(1)             # [B, 1792]
        x = self.projector(x)        # [B, 256]
        return x


# ============================================================
# BLOCK 2: CONVFORMER MODULE
# Models global spatial relationships across tissue regions:
# lobular fat distribution, zonal steatosis patterns
# ============================================================
class ConvPatchEmbed(nn.Module):
    """
    Convolutional patch embedding for ConvFormer.
    Replaces standard linear projection with a strided
    convolution to preserve local spatial structure
    during tokenization — critical for histopathology
    where spatial arrangement carries diagnostic meaning.

    Input:  [B, 3, 256, 256]
    Output: [B, num_patches, embed_dim]
    """
    def __init__(self, img_size=INPUT_SIZE, patch_size=16,
                 in_channels=3, embed_dim=EMBED_DIM):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Sequential(
            # Two-stage conv embedding: preserves local structure
            nn.Conv2d(in_channels, embed_dim // 2,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim,
                      kernel_size=3, stride=patch_size // 2,
                      padding=1),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x):
        x = self.proj(x)             # [B, embed_dim, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x, H, W


class ConvFormerBlock(nn.Module):
    """
    Single ConvFormer block combining:
        1. Depthwise convolution — local inductive bias
           (preserves spatial locality within attention)
        2. Multi-head self-attention — global context
        3. MLP with GELU activation — feature transformation

    The depthwise conv before attention is the key
    distinction from a standard ViT block — it provides
    translational equivariance beneficial for tissue
    pattern recognition.
    """
    def __init__(self, dim=EMBED_DIM, num_heads=NUM_HEADS,
                 mlp_ratio=MLP_RATIO, dropout=DROPOUT_RATE):
        super().__init__()

        self.norm1   = nn.LayerNorm(dim)
        self.norm2   = nn.LayerNorm(dim)

        # Depthwise conv for local inductive bias
        self.dw_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3,
                      padding=1, groups=dim),
            nn.GELU()
        )

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # MLP block
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # --- Local conv branch ---
        # x: [B, N, C] → transpose for Conv1d → [B, C, N]
        conv_out = self.dw_conv(
            self.norm1(x).transpose(1, 2)
        ).transpose(1, 2)             # back to [B, N, C]

        # --- Self-attention branch ---
        attn_in  = x + conv_out
        attn_out, _ = self.attn(
            attn_in, attn_in, attn_in
        )
        x = attn_in + attn_out        # residual

        # --- MLP branch ---
        x = x + self.mlp(self.norm2(x))
        return x


class ConvFormerBranch(nn.Module):
    """
    Full ConvFormer branch:
        Patch embedding → N × ConvFormerBlock → Global pooling

    Processes the full 256×256 tile as a sequence of
    patch tokens, modeling long-range dependencies
    between tissue regions that CNN receptive fields
    cannot capture.

    Output:
        Tensor of shape [B, 256] — global context vector
    """
    def __init__(self, img_size=INPUT_SIZE, patch_size=16,
                 embed_dim=EMBED_DIM, depth=DEPTH,
                 num_heads=NUM_HEADS):
        super().__init__()

        self.patch_embed = ConvPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        self.blocks = nn.Sequential(*[
            ConvFormerBlock(
                dim=embed_dim,
                num_heads=num_heads
            )
            for _ in range(depth)
        ])

        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x):
        x, H, W  = self.patch_embed(x)    # [B, N, embed_dim]
        x        = self.blocks(x)         # [B, N, embed_dim]
        x        = self.norm(x)
        x        = x.mean(dim=1)          # global average pool
        x        = self.dropout(x)        # [B, embed_dim]
        return x


# ============================================================
# BLOCK 3: ATTENTION FUSION MODULE
# Dynamically weights CNN local features vs.
# ConvFormer global features per input tile
# ============================================================
class AttentionFusion(nn.Module):
    """
    Soft attention-based fusion of CNN and ConvFormer
    feature vectors.

    Rather than naive concatenation, a learned gating
    mechanism produces per-branch weights, allowing
    the model to rely more on local features for
    Grade 0/1 tiles (subtle fat changes) and more on
    global context for Grade 2/3 tiles (extensive
    lobular involvement).

    Fusion formula:
        w = softmax(W · [f_cnn; f_convformer])
        f_fused = w[0] * f_cnn + w[1] * f_convformer

    Output:
        Tensor of shape [B, 256]
    """
    def __init__(self, dim=EMBED_DIM):
        super().__init__()

        # Gate: maps concatenated features to 2 scalar weights
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, f_cnn, f_convformer):
        # Compute attention weights
        combined = torch.cat([f_cnn, f_convformer], dim=-1)
        weights  = self.gate(combined)           # [B, 2]

        w_cnn    = weights[:, 0].unsqueeze(-1)   # [B, 1]
        w_conv   = weights[:, 1].unsqueeze(-1)   # [B, 1]

        # Weighted combination
        fused = w_cnn * f_cnn + w_conv * f_convformer
        return fused, weights


# ============================================================
# FULL MODEL: HybridCNNConvFormer
# ============================================================
class HybridCNNConvFormer(nn.Module):
    """
    Hybrid CNN–ConvFormer model for steatosis grade
    classification from H&E histopathology tiles.

    Architecture:
        ┌─────────────┐    ┌──────────────────┐
        │ CNN Backbone│    │ ConvFormer Branch │
        │ (EfficientNet│   │ (Patch Embed +    │
        │  -B4)        │   │  4× ConvFormer    │
        │              │   │  Blocks)          │
        └──────┬───────┘   └────────┬─────────┘
               │                    │
               └────────┬───────────┘
                    [Attention Fusion]
                         │
                  [Classification Head]
                         │
                  4-class softmax output
                  (Grade 0 / 1 / 2 / 3)

    Parameters:
        num_classes (int): Number of output classes. Default 4.
        pretrained  (bool): Use ImageNet weights for CNN. Default True.
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()

        self.cnn_branch        = CNNBackbone(
            out_dim=EMBED_DIM,
            pretrained=pretrained
        )
        self.convformer_branch = ConvFormerBranch(
            embed_dim=EMBED_DIM,
            depth=DEPTH,
            num_heads=NUM_HEADS
        )
        self.fusion            = AttentionFusion(dim=EMBED_DIM)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Extract features from both branches
        f_cnn   = self.cnn_branch(x)          # [B, 256]
        f_conv  = self.convformer_branch(x)   # [B, 256]

        # Fuse with attention gating
        fused, fusion_weights = self.fusion(
            f_cnn, f_conv
        )                                     # [B, 256]

        # Classify
        logits = self.classifier(fused)       # [B, 4]

        return logits, fusion_weights


# ============================================================
# MODEL SUMMARY UTILITY
# ============================================================
def print_model_summary(model, input_size=(1, 3, 256, 256)):
    """
    Prints parameter counts per component.
    Reports in §3.6 and §3.10.
    """
    def count_params(module):
        return sum(p.numel() for p in module.parameters()
                   if p.requires_grad)

    total     = count_params(model)
    cnn_p     = count_params(model.cnn_branch)
    conv_p    = count_params(model.convformer_branch)
    fusion_p  = count_params(model.fusion)
    head_p    = count_params(model.classifier)

        
    print("=== MODEL SUMMARY ===")
    print(f"  CNN Backbone        : {cnn_p:>12,} params")
    print(f"  ConvFormer Branch   : {conv_p:>12,} params")
    print(f"  Attention Fusion    : {fusion_p:>12,} params")
    print(f"  Classification Head : {head_p:>12,} params")
    print(f"  {'─'*38}")
    print(f"  TOTAL               : {total:>12,} params")

    # Forward pass test
    # device = next(model.parameters()).device
    # dummy  = torch.randn(input_size).to(device)
    # with torch.no_grad():
    #     logits, weights = model(dummy)

    device = next(model.parameters()).device
    model.eval()  # IMPORTANT: disables BatchNorm/Dropout training behavior
    dummy  = torch.randn(input_size).to(device)

    with torch.no_grad():
        logits, weights = model(dummy)

    print(f"\n  Input  shape : {tuple(dummy.shape)}")
    print(f"  Output shape : {tuple(logits.shape)}")
    print(f"  Fusion weights (sample): "
          f"CNN={weights[0,0].item():.3f}, "
          f"ConvFormer={weights[0,1].item():.3f}")

    with open("buildFiles/model_summary.txt", "w") as f:
        f.write("=== MODEL SUMMARY ===\n")
        f.write('Device: ' + str(device) + '\n\n')
        f.write(f"  CNN Backbone        : {cnn_p:,} params\n")
        f.write(f"  ConvFormer Branch   : {conv_p:,} params\n")
        f.write(f"  Attention Fusion    : {fusion_p:,} params\n")
        f.write(f"  Classification Head : {head_p:,} params\n")
        f.write(f"  TOTAL               : {total:,} params\n")
        f.write(f"\n  Input  shape : {tuple(dummy.shape)}\n")
        f.write(f"  Output shape : {tuple(logits.shape)}\n")
        f.write(f"  Fusion weights (sample): "
                f"CNN={weights[0,0].item():.3f}, "
                f"ConvFormer={weights[0,1].item():.3f}\n")


# ============================================================
# ENTRY POINT — ARCHITECTURE VERIFICATION
# ============================================================
if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}\n")

    model = HybridCNNConvFormer(
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(device)

    print_model_summary(model)
    