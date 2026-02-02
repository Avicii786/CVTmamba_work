import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableHyperCorrelationSCM(nn.Module):
    """
    SOTA Upgrade: Uncertainty-Aware SCM.
    Predicts an 'Occlusion/Validity Mask' alongside flow to handle 
    appearing/disappearing objects (e.g., Building -> Empty).
    """
    def __init__(self, in_channels, search_radius=4):
        super().__init__()
        self.in_channels = in_channels
        self.search_radius = search_radius
        self.kernel_size = 2 * search_radius + 1
        
        # GroupNorm is more stable than BatchNorm for small batch sizes (1 or 2)
        gn_groups = 32 if in_channels >= 32 else 4
        gn_groups_reduced = 32 if (in_channels // 2) >= 32 else 4

        # [UPGRADE] Predict 3 channels: Flow (2) + Validity Mask (1)
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, in_channels), 
            nn.SiLU(inplace=True), # SiLU is preferred for Mamba-based architectures
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups_reduced, in_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // 2, 3, kernel_size=3, padding=1, bias=True) # 2 Flow + 1 Mask
        )
        
        self.corr_dim = self.kernel_size ** 2
        
        self.corr_proj = nn.Sequential(
            nn.Conv2d(self.corr_dim, in_channels // 2, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups_reduced, in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.diff_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups_reduced, in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        )
        
        self.lambda_proj = nn.Sequential(
             nn.Conv2d(in_channels, 1, kernel_size=1, bias=True),
             nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize flow to 0
        last_conv = self.offset_predictor[-1]
        nn.init.constant_(last_conv.weight, 0)
        nn.init.constant_(last_conv.bias, 0)
        
        # [CRITICAL] Bias the mask channel (index 2) to be positive initially.
        # This ensures the model starts by trusting the alignment (mask ~ 1.0)
        # and learns to suppress specific regions later.
        with torch.no_grad():
            last_conv.bias[2] = 2.0 

    def compute_cost_volume(self, feat1, feat2):
        B, C, H, W = feat1.shape
        padded_feat2 = F.pad(feat2, [self.search_radius]*4, mode='constant', value=0)
        
        cost_volume = []
        for y in range(self.kernel_size):
            for x in range(self.kernel_size):
                feat2_slice = padded_feat2[:, :, y:y+H, x:x+W]
                corr = (feat1 * feat2_slice).mean(dim=1, keepdim=True) 
                cost_volume.append(corr)
                
        return torch.cat(cost_volume, dim=1)

    def warp(self, x, flow):
        B, C, H, W = x.shape
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        
        grid = torch.cat((xx, yy), 1).float().to(x.device)
        vgrid = grid + flow
        
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return output

    def forward(self, feat1, feat2):
        flow_input = torch.cat([feat1, feat2], dim=1)
        
        # Predict Flow (channels 0,1) and Validity Mask (channel 2)
        prediction = self.offset_predictor(flow_input)
        flow = prediction[:, :2, :, :]
        mask_logits = prediction[:, 2:3, :, :]
        
        # Validity Mask: 1.0 = Valid Match, 0.0 = Occluded/Changed
        validity_mask = torch.sigmoid(mask_logits) 
        
        # Warp feat2 to align with feat1
        feat2_warped = self.warp(feat2, flow)
        
        # [UPGRADE] Apply the validity mask. 
        # Suppress features where the model thinks the object is missing in T2.
        feat2_aligned = feat2_warped * validity_mask
        
        cost_vol = self.compute_cost_volume(feat1, feat2_aligned)
        diff_map = torch.abs(feat1 - feat2_aligned) 
        
        cost_emb = self.corr_proj(cost_vol)
        diff_emb = self.diff_proj(diff_map)
        
        fusion_in = torch.cat([cost_emb, diff_emb], dim=1)
        fused_feat = self.fusion(fusion_in) 
        
        lambda_map = self.lambda_proj(fused_feat)

        return feat2_aligned, flow, lambda_map