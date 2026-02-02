import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CTVMambaBackbone 
from .scm import DeformableHyperCorrelationSCM
# [CHANGE] Import EfficientCTR instead of DynamicGraphRefiner
from .ctr import EfficientCTR
from .decoder import MultiScaleMambaDecoder, BCDDecoder

MODEL_CONFIGS = {
    'tiny': {
        'embed_dims': [96, 192, 384, 768],
        'depths': [2, 2, 9, 2],
        'drop_rate': 0.0,
    },
    'small': {
        'embed_dims': [96, 192, 384, 768],
        'depths': [2, 2, 27, 2],
        'drop_rate': 0.1,
    },
    'base': {
        'embed_dims': [128, 256, 512, 1024],
        'depths': [2, 2, 27, 2],
        'drop_rate': 0.1,
    }
}

class LambdaSCD(nn.Module):
    """
    Lambda-SCD v3.0 (Lightweight SOTA): 
    - Symmetric Alignment (Solves Relief Displacement Misalignment)
    - Pyramid Warping (Solves Registration Noise in BCD)
    - Efficient Coordinate Attention (Replaces Heavy Graph Refiner)
    - Lightweight Decoders
    - Uncertainty-Aware SCM
    """
    def __init__(self, in_channels=3, num_classes=7, model_type='tiny', **kwargs):
        super().__init__()
        
        if model_type not in MODEL_CONFIGS: config = MODEL_CONFIGS['tiny']
        else: config = MODEL_CONFIGS[model_type]
            
        embed_dims = kwargs.get('embed_dims', config['embed_dims'])
        depths = kwargs.get('depths', config['depths'])
        drop_rate = kwargs.get('drop_rate', config['drop_rate'])
        
        self.num_classes = num_classes
        
        # 1. Backbone (CT-VMamba)
        self.backbone = CTVMambaBackbone(
            in_chans=in_channels, embed_dims=embed_dims, depths=depths, drop_rate=drop_rate
        )

        # 2. SCM (Siamese Alignment)
        # We use one SCM instance to learn alignment in both directions (Regularization)
        # SCM now returns [Aligned Feat, Flow, Validity Mask (Lambda)]
        self.scm = DeformableHyperCorrelationSCM(in_channels=embed_dims[-1])

        # 3. CTR (Efficient Coordinate Refiner)
        # [CHANGE] Replaced DynamicGraphRefiner with EfficientCTR
        # This refines the *features* using coordinate attention (O(N)) instead of 
        # refining the lambda map with a graph (O(N^2)).
        self.ctr = EfficientCTR(
            in_channels=embed_dims[-1], 
            reduction=16
        )

        # 4. Decoders (Tri-Branch)
        self.sem_decoder = MultiScaleMambaDecoder(embed_dims, num_classes)
        self.bcd_decoder = BCDDecoder(embed_dims)

    def _align_feature_pyramid(self, feats, flow):
        """
        Warps a list of multi-scale features using the learned flow field.
        Used to align the T2 pyramid to T1 before BCD.
        """
        aligned_feats = []
        for feat in feats:
            B, C, H, W = feat.shape
            
            # 1. Handle Resolution Mismatch
            if flow.shape[-2:] != (H, W):
                scale_factor = H / flow.shape[2] 
                
                # Interpolate Flow
                flow_curr = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
                
                # Scale Magnitude (Pixel displacement grows with resolution)
                flow_curr = flow_curr * scale_factor
            else:
                flow_curr = flow
                
            # 2. Warp using SCM's built-in warping logic
            feat_warped = self.scm.warp(feat, flow_curr)
            aligned_feats.append(feat_warped)
            
        return aligned_feats

    def forward(self, img1, img2):
        # 1. Feature Extraction
        # feats_t1/t2 are lists: [1/4, 1/8, 1/16, 1/32]
        feats_t1, feats_t2 = self.backbone(img1, img2)
        deep_t1 = feats_t1[-1]
        deep_t2 = feats_t2[-1]

        # 2. Symmetric Alignment
        
        # A) Forward: Align T2 -> T1
        # flow_fwd captures the relief displacement needed to move T2 onto T1
        # lambda_t1 is the "Validity Mask" (1=Valid, 0=Occluded/Changed)
        aligned_deep_t2, flow_fwd, lambda_t1 = self.scm(deep_t1, deep_t2)
        
        # B) Reverse: Align T1 -> T2
        # We need lambda_t2 for the T2 semantic branch
        _, _, lambda_t2 = self.scm(deep_t2, deep_t1)
        
        # 3. CTR Refinement (Efficient Feature Refinement)
        # [CHANGE] Now we refine the FEATURES directly using Coordinate Attention.
        # This enhances global context in the deepest features before decoding.
        deep_t1_refined = self.ctr(deep_t1)
        aligned_deep_t2_refined = self.ctr(aligned_deep_t2)
        
        # Update the pyramids with refined deep features
        feats_t1[-1] = deep_t1_refined
        feats_t2[-1] = aligned_deep_t2_refined # Note: This is now aligned AND refined

        # 4. Decoding
        
        # A) Semantic Decoder T1
        # Inputs: Refined T1 Pyramid + Lambda T1 Mask
        logits_t1, kld_t1 = self.sem_decoder(feats_t1, lambda_map=lambda_t1)
        
        # B) Semantic Decoder T2
        # For T2, we use the original T2 pyramid (with refined deep layer) and T2 lambda
        # Note: We create a temporary pyramid for T2 decoding that uses the refined DEEP T2 (unaligned for semantic)
        # Actually, for semantic T2, we want to predict in T2 space.
        # We need the UNALIGNED but REFINED T2 deep feature.
        # Let's re-refine the unaligned T2 deep feature for the semantic branch to be correct.
        deep_t2_refined = self.ctr(deep_t2)
        feats_t2_sem = list(feats_t2) # Shallow copy
        feats_t2_sem[-1] = deep_t2_refined
        
        logits_t2, kld_t2 = self.sem_decoder(feats_t2_sem, lambda_map=lambda_t2)
        
        # C) Binary Change Decoder
        # To compute clean differences for BCD, we warp the ENTIRE T2 pyramid to match T1
        # We use the refined aligned T2 deep feature we computed earlier for the deep level
        feats_t2_for_bcd = list(feats_t2)
        feats_t2_for_bcd[-1] = aligned_deep_t2_refined # already aligned and refined
        
        # Warp the rest of the pyramid
        aligned_feats_t2_pyramid = self._align_feature_pyramid(feats_t2_for_bcd[:-1], flow_fwd)
        aligned_feats_t2_pyramid.append(aligned_deep_t2_refined)
        
        # Inputs: Refined T1 + Aligned/Refined T2 + Lambda T1
        logits_bcd = self.bcd_decoder(feats_t1, aligned_feats_t2_pyramid, lambda_map=lambda_t1)
        
        # 5. Upsample
        H, W = img1.shape[2], img1.shape[3]
        out_sem1 = F.interpolate(logits_t1, size=(H, W), mode='bilinear', align_corners=False)
        out_sem2 = F.interpolate(logits_t2, size=(H, W), mode='bilinear', align_corners=False)
        out_bcd  = F.interpolate(logits_bcd, size=(H, W), mode='bilinear', align_corners=False)

        return {
            "sem1": out_sem1,
            "sem2": out_sem2,
            "bcd": out_bcd,
            "flow": flow_fwd,
            "lambda_t2": lambda_t2,
            "kld_loss": kld_t1 + kld_t2 # Aggregate KLD loss from both streams
        }