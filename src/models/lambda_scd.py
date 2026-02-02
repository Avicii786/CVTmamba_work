import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CTVMambaBackbone 
from .scm import DeformableHyperCorrelationSCM
from .ctr import EfficientCTR
# [CHANGE] Import the new MLP Decoders
from .decoder import SegFormerHead, LightweightBCDDecoder

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
    Lambda-SCD v4.0 (SegFormer-SCD): 
    - Backbone: CT-VMamba (Global Context)
    - Alignment: SCM
    - Refiner: Efficient CTR
    - Decoder: All-MLP SegFormer Head (Lightweight SOTA)
    """
    def __init__(self, in_channels=3, num_classes=7, model_type='tiny', **kwargs):
        super().__init__()
        
        if model_type not in MODEL_CONFIGS: config = MODEL_CONFIGS['tiny']
        else: config = MODEL_CONFIGS[model_type]
            
        embed_dims = kwargs.get('embed_dims', config['embed_dims'])
        depths = kwargs.get('depths', config['depths'])
        drop_rate = kwargs.get('drop_rate', config['drop_rate'])
        
        self.num_classes = num_classes
        
        # 1. Backbone
        self.backbone = CTVMambaBackbone(
            in_channels=in_channels, embed_dims=embed_dims, depths=depths, drop_rate=drop_rate
        )

        # 2. SCM (Alignment)
        self.scm = DeformableHyperCorrelationSCM(in_channels=embed_dims[-1])

        # 3. CTR (Refiner)
        self.ctr = EfficientCTR(in_channels=embed_dims[-1], reduction=16)

        # 4. Decoders (All-MLP)
        # We use a common embedding dim of 128 for the decoder (very light)
        decoder_dim = 128
        self.sem_decoder = SegFormerHead(embed_dims, embedding_dim=decoder_dim, num_classes=num_classes)
        
        # BCD Decoder now just looks at the fused representations
        self.bcd_decoder = LightweightBCDDecoder(embedding_dim=decoder_dim)

    def _align_feature_pyramid(self, feats, flow):
        # ... (Same as before, simplified for brevity in this specific file if needed, 
        # but actually for SegFormer head we might not need to warp the whole pyramid 
        # explicitly if we align at the fusion stage. However, keeping it for robustness.)
        aligned_feats = []
        for feat in feats:
            B, C, H, W = feat.shape
            if flow.shape[-2:] != (H, W):
                scale_factor = H / flow.shape[2] 
                flow_curr = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False) * scale_factor
            else:
                flow_curr = flow
            aligned_feats.append(self.scm.warp(feat, flow_curr))
        return aligned_feats

    def forward_single_stream(self, feats, lambda_map, flow=None, align=False):
        """
        Helper to run the SegFormer Head.
        Returns: logits, aux_outputs, kld_loss, fused_embedding
        """
        # If alignment is requested (for T2), we warp the pyramid first
        if align and flow is not None:
            feats = self._align_feature_pyramid(feats, flow)
            
        # SegFormer Decoder internally fuses multiscale features
        # We modify SegFormerHead slightly to return the 'fused' feature for BCD usage
        # (See modified forward call below logic)
        
        # Let's call the decoder
        # Note: We need to access the intermediate 'fused' feature for BCD.
        # Ideally, we'd refactor SegFormerHead to return it.
        # For now, let's just rely on the logits or assume SegFormerHead returns it.
        
        # To make this clean without changing decoder.py interface too much:
        # We will just run the decoder.
        pass

    def forward(self, img1, img2):
        # 1. Feature Extraction
        feats_t1, feats_t2 = self.backbone(img1, img2)
        deep_t1 = feats_t1[-1]
        deep_t2 = feats_t2[-1]

        # 2. Alignment & Refinement
        aligned_deep_t2, flow_fwd, lambda_t1 = self.scm(deep_t1, deep_t2)
        _, _, lambda_t2 = self.scm(deep_t2, deep_t1)
        
        deep_t1 = self.ctr(deep_t1)
        aligned_deep_t2 = self.ctr(aligned_deep_t2)
        
        feats_t1[-1] = deep_t1
        feats_t2[-1] = aligned_deep_t2 # This T2 is aligned to T1

        # 3. Semantic Decoding (T1)
        logits_t1, aux_t1, kld_t1 = self.sem_decoder(feats_t1, lambda_map=lambda_t1)
        
        # 4. Semantic Decoding (T2)
        # For T2 Semantic, we want predictions in T2's original geometry.
        # So we use the original (unaligned) T2 features.
        # But we want to use the refined deep feature... which we calculated as 'aligned_deep_t2'.
        # We need to re-refine the UNALIGNED T2 for the semantic branch.
        # This small overhead is worth it for correctness.
        deep_t2_native = self.ctr(deep_t2) 
        feats_t2_native = list(feats_t2)
        feats_t2_native[-1] = deep_t2_native
        
        logits_t2, aux_t2, kld_t2 = self.sem_decoder(feats_t2_native, lambda_map=lambda_t2)

        # 5. BCD Decoding
        # Strategy: Use the Penultimate Features (Fused Embeddings) from the semantic decoder?
        # Or just use the features?
        # Let's use the features.
        # T1 is already ready: feats_t1
        # T2 needs to be fully aligned to T1 for BCD.
        # We aligned the deep layer, but let's align the rest of the pyramid efficiently.
        aligned_feats_t2 = self._align_feature_pyramid(feats_t2_native[:-1], flow_fwd)
        aligned_feats_t2.append(aligned_deep_t2) # The deep one is already aligned
        
        # Now we need a lightweight way to compare feats_t1 and aligned_feats_t2.
        # We can run the SegFormer fusion on both, then compare the fused embeddings.
        
        # Hack to get fused embeddings from SegFormerHead without changing its return sig drastically:
        # We just instantiate a dedicated "FeatureAggregator" for BCD which is just the first half of SegFormer.
        # OR: We trust that the SemDecoder learns good features and we reuse them.
        
        # Let's extract fused features by manually running the projection part of SemDecoder
        # (This shares weights with SemDecoder!)
        with torch.no_grad(): # Optional: Detach BCD from Semantic gradients? No, keep connected.
             pass

        # To keep it simple and runnable:
        # We run the SemDecoder projection logic manually here to get the embeddings.
        # (Ideally refactor SegFormerHead to separate 'encode' and 'predict')
        
        # Helper to fuse
        def fuse_feats(feats):
            outs = []
            H_target, W_target = feats[0].shape[2], feats[0].shape[3]
            for i, feat in enumerate(feats):
                B, C, H, W = feat.shape
                feat_flat = feat.flatten(2).transpose(1, 2)
                embed = self.sem_decoder.linear_layers[i](feat_flat) # Reuse weights
                embed = embed.transpose(1, 2).reshape(B, -1, H, W)
                if i > 0:
                    embed = F.interpolate(embed, size=(H_target, W_target), mode='bilinear', align_corners=False)
                outs.append(embed)
            return self.sem_decoder.fusion(torch.cat(outs, dim=1)) # Reuse weights

        fused_t1 = fuse_feats(feats_t1)
        fused_t2 = fuse_feats(aligned_feats_t2)
        
        logits_bcd = self.bcd_decoder(fused_t1, fused_t2)
        
        # 6. Upsample Results
        H, W = img1.shape[2], img1.shape[3]
        out_sem1 = F.interpolate(logits_t1, size=(H, W), mode='bilinear', align_corners=False)
        out_sem2 = F.interpolate(logits_t2, size=(H, W), mode='bilinear', align_corners=False)
        out_bcd  = F.interpolate(logits_bcd, size=(H, W), mode='bilinear', align_corners=False)

        return {
            "sem1": out_sem1,
            "sem2": out_sem2,
            "sem1_aux": aux_t1,
            "sem2_aux": aux_t2,
            "bcd": out_bcd,
            "flow": flow_fwd,
            "kld_loss": kld_t1 + kld_t2
        }
