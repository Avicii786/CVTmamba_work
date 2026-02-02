import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientCTR(nn.Module):
    """
    [SOTA UPGRADE] Efficient Contextual Temporal Refiner.
    Replaces heavy Dynamic Graph (O(N^2)) with Coordinate Attention (O(N)).
    
    Mechanism:
    Instead of building a massive attention map between every pixel,
    we pool features along the Height and Width axes independently.
    This captures long-range dependencies (context) without the memory explosion.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # Squeeze the channels for efficiency (Bottleneck)
        mip = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.GroupNorm(4, mip) # GroupNorm is more stable than BN for small batches
        self.act = nn.Hardswish() # Modern activation function (MobileNetV3/SOTA)
        
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 1. Coordinate Pooling
        # X_H: [B, C, H, 1]
        # X_W: [B, C, 1, W]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # Permute to [B, C, W, 1] for concatenation
        
        # 2. Shared Transformation
        # Concatenate along spatial dimension to process H and W correlation together
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # 3. Split and Expand
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2) # Restore to [B, C, 1, W]
        
        # 4. Generate Attention Weights
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        # 5. Refine Features
        # Reweight input features based on vertical and horizontal context
        out = identity * a_h * a_w
        
        return out