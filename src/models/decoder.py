import torch
import torch.nn as nn
import torch.nn.functional as F

class IBGate(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid_channels = in_channels // reduction
        self.gate_encoder = nn.Sequential(
            nn.Conv2d(in_channels + 1, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels * 2, kernel_size=1),
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, lambda_map):
        if lambda_map.shape[-2:] != x.shape[-2:]:
            gate_map = F.interpolate(lambda_map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        else:
            gate_map = lambda_map

        gate_params = self.gate_encoder(torch.cat([x, gate_map], dim=1))
        mu, logvar = gate_params.chunk(2, dim=1)
        
        # Stability Clamp
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        z = self.reparameterize(mu, logvar) if self.training else mu
        gate = torch.sigmoid(z)
        
        kld_element = 1 + logvar - mu.pow(2) - logvar.exp()
        kld_loss = -0.5 * torch.mean(kld_element)
        
        return x * (1 + gate), kld_loss

class FeatureDifferenceModule(nn.Module):
    """
    Asymmetric Difference Module.
    """
    def __init__(self, in_channels):
        super().__init__()
        num_groups = 32 if in_channels >= 32 else 4
        # Input channels = in_channels * 3 (T1, T2, T1-T2)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, in_channels), 
            nn.ReLU(inplace=True)
        )
        
    def forward(self, f1, f2):
        return self.fusion(torch.cat([f1, f2, f1 - f2], dim=1))

class LightweightFusionBlock(nn.Module):
    """
    [LIGHTWEIGHT REPLACEMENT]
    Replaces the heavy MambaFusionBlock with a standard Residual Conv Block.
    The Encoder (Mamba) already did the global context heavy lifting.
    The Decoder just needs to fuse and refine locally.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.skip_proj = nn.Conv2d(skip_channels, in_channels, kernel_size=1, bias=False)
        
        num_groups = 32 if out_channels >= 32 else 4
        
        # ResNet-style Residual Block
        self.conv1 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups, out_channels)
        
        self.downsample = None
        if in_channels * 2 != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups, out_channels)
            )

    def forward(self, x, skip):
        x_up = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        skip_proj = self.skip_proj(skip)
        
        out = torch.cat([x_up, skip_proj], dim=1)
        residual = out
        
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        out += residual
        return self.relu(out)

class MultiScaleMambaDecoder(nn.Module):
    def __init__(self, encoder_dims, num_classes):
        super().__init__()
        # [UPGRADE] Using LightweightFusionBlock
        self.stage1 = LightweightFusionBlock(encoder_dims[3], encoder_dims[2], encoder_dims[2])
        self.stage2 = LightweightFusionBlock(encoder_dims[2], encoder_dims[1], encoder_dims[1])
        self.stage3 = LightweightFusionBlock(encoder_dims[1], encoder_dims[0], encoder_dims[0])
        
        self.ib_gate = IBGate(encoder_dims[3])
        self.head = nn.Conv2d(encoder_dims[0], num_classes, kernel_size=1)
        self.aux1 = nn.Conv2d(encoder_dims[2], num_classes, kernel_size=1) 
        self.aux2 = nn.Conv2d(encoder_dims[1], num_classes, kernel_size=1)

    def forward(self, features, lambda_map=None):
        c1, c2, c3, c4 = features
        x, kld = self.ib_gate(c4, lambda_map) if lambda_map is not None else (c4, torch.tensor(0.0, device=c1.device))
        x = self.stage1(x, c3)
        a1 = self.aux1(x)
        x = self.stage2(x, c2)
        a2 = self.aux2(x)
        x = self.stage3(x, c1)
        return self.head(x), [a1, a2], kld

class BCDDecoder(nn.Module):
    def __init__(self, encoder_dims):
        super().__init__()
        self.diff_extractors = nn.ModuleList([FeatureDifferenceModule(dim) for dim in encoder_dims])
        # [UPGRADE] Using LightweightFusionBlock
        self.stage1 = LightweightFusionBlock(encoder_dims[3], encoder_dims[2], encoder_dims[2])
        self.stage2 = LightweightFusionBlock(encoder_dims[2], encoder_dims[1], encoder_dims[1])
        self.stage3 = LightweightFusionBlock(encoder_dims[1], encoder_dims[0], encoder_dims[0])
        self.head = nn.Conv2d(encoder_dims[0], 1, kernel_size=1)
        
    def forward(self, feats_t1, feats_t2, lambda_map):
        diffs = [self.diff_extractors[i](f1, f2) for i, (f1, f2) in enumerate(zip(feats_t1, feats_t2))]
        x = self.stage1(diffs[3], diffs[2])
        x = self.stage2(x, diffs[1])
        x = self.stage3(x, diffs[0])
        return self.head(x)