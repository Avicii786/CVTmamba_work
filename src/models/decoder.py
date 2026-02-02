import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Simple MLP block: Linear -> ReLU -> Linear
    Literature: SegFormer (NeurIPS 2021)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)

class SegFormerHead(nn.Module):
    """
    SegFormer-style All-MLP Decoder.
    1. Projects multi-scale features to common dimension C.
    2. Upsamples all to 1/4 resolution.
    3. Concatenates -> 4*C.
    4. Predicts class logits.
    
    Why SOTA?
    - Removes heavy convolutions/attention from decoder.
    - Relies on the Encoder (Mamba) for global context.
    - Extremely fast and memory efficient.
    """
    def __init__(self, encoder_dims, embedding_dim=128, num_classes=7):
        super().__init__()
        self.linear_layers = nn.ModuleList()
        self.scaling_factor = 4 # Upsample to 1/4 resolution
        
        # 1. MLP Projections for each scale
        for dim in encoder_dims:
            self.linear_layers.append(MLP(dim, embedding_dim))

        # 2. Fusion Layer
        # Input: 4 scales * embedding_dim
        self.fusion = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.GroupNorm(16, embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # 3. Classifier Head
        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        
        # 4. Aux Head (Optional Deep Supervision on deep features)
        self.aux_head = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features, lambda_map=None):
        # features list: [c1, c2, c3, c4]
        # c1: 1/4, c2: 1/8, c3: 1/16, c4: 1/32
        
        batch_size = features[0].shape[0]
        
        # Target size is 1/4 resolution (shape of c1)
        H_target, W_target = features[0].shape[2], features[0].shape[3]
        
        outs = []
        for i, feat in enumerate(features):
            # Flatten: [B, C, H, W] -> [B, H*W, C] -> MLP -> [B, C, H, W]
            B, C, H, W = feat.shape
            
            # Apply MLP (Linear layer works on last dim)
            feat_flat = feat.flatten(2).transpose(1, 2) # [B, L, C]
            embed = self.linear_layers[i](feat_flat)
            embed = embed.transpose(1, 2).reshape(B, -1, H, W) # Back to [B, Emb, H, W]
            
            # Upsample to common resolution (1/4)
            if i > 0:
                embed = F.interpolate(embed, size=(H_target, W_target), mode='bilinear', align_corners=False)
            
            outs.append(embed)
            
        # Concatenate: [B, 4*Emb, H/4, W/4]
        concat_feats = torch.cat(outs, dim=1)
        
        # Fuse
        fused = self.fusion(concat_feats)
        
        # Apply Information Bottleneck (IB) gating if provided
        kld_loss = torch.tensor(0.0, device=fused.device)
        if lambda_map is not None:
            # Resize lambda to 1/4 if needed
            if lambda_map.shape[-2:] != fused.shape[-2:]:
                lambda_map = F.interpolate(lambda_map, size=fused.shape[-2:], mode='bilinear', align_corners=False)
            
            # Gate features: Multiply by validity mask (1=Valid, 0=Changed/Occluded)
            fused = fused * lambda_map 
            
            # (Optional) Simple KLD regularization on the fusion magnitude could go here, 
            # but for MLP decoder we usually skip complex VAE logic to keep it light.
            
        # Prediction
        logits = self.classifier(fused)
        
        # Aux output (from the deepest feature, upsampled)
        aux_out = self.aux_head(outs[-1]) 
        
        return logits, [aux_out], kld_loss

class LightweightBCDDecoder(nn.Module):
    """
    Computes Binary Change from the fused Semantic Features.
    Difference -> MLP -> Logits.
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Input is 2 * embedding_dim (concatenated T1 and T2 fused features)
        self.proj = nn.Sequential(
            nn.Conv2d(embedding_dim * 2, embedding_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim, 1, kernel_size=1)
        )
        
    def forward(self, fused_t1, fused_t2):
        # We assume fused_t1 and fused_t2 are the outputs of the SegFormer Fusion layer
        # They are already "Global Context Enriched"
        
        # Concatenate T1 and T2
        x = torch.cat([fused_t1, fused_t2], dim=1)
        logits = self.proj(x)
        return logits
