import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowSmoothnessLoss(nn.Module):
    def forward(self, flow):
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        return (torch.mean(dx) + torch.mean(dy))

class SemanticConsistencyLoss(nn.Module):
    def __init__(self, metric='mse'):
        super().__init__()
        self.metric = metric

    def forward(self, logits_t1, logits_t2, change_mask):
        valid_pixels = (change_mask != 255)
        unchanged_mask = (change_mask == 0) & valid_pixels
        
        if unchanged_mask.sum() < 1e-5:
            return torch.tensor(0.0, device=logits_t1.device)

        prob_t1 = F.softmax(logits_t1, dim=1)
        prob_t2 = F.softmax(logits_t2, dim=1)
        
        mse = F.mse_loss(prob_t1, prob_t2, reduction='none').mean(dim=1)
        loss = (mse * unchanged_mask).sum() / (unchanged_mask.sum() + 1e-6)
        return loss

class FocalLoss(nn.Module):
    """
    SOTA Upgrade: Multi-class Focal Loss for handling severe class imbalance.
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # Calculate standard Cross Entropy first
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        
        # Focal term: (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        preds = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        valid_mask = (targets != self.ignore_index)
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        
        if preds.numel() == 0:
            return torch.tensor(0.0, device=logits.device)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice

class LambdaSCDLoss(nn.Module):
    def __init__(self, num_classes=7, ignore_index=255, lambda_scl=1.0, lambda_kld=0.1, lambda_flow=0.01):
        super().__init__()
        self.ignore_index = ignore_index
        self.lambda_scl = lambda_scl
        self.lambda_kld = lambda_kld
        self.lambda_flow = lambda_flow
        
        # [UPGRADE] Replaced hardcoded weights with Focal Loss
        self.sem_focal = FocalLoss(gamma=2.0, ignore_index=ignore_index)
        
        # Binary Change Detection Losses
        self.bcd_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))
        self.bcd_dice = DiceLoss(ignore_index=ignore_index)
        
        # Auxiliary Losses
        self.scl_loss = SemanticConsistencyLoss(metric='mse') 
        self.flow_loss = FlowSmoothnessLoss()

    def calc_sem_loss(self, logits, targets):
        return self.sem_focal(logits, targets)

    def forward(self, outputs, targets):
        loss_sem1 = self.calc_sem_loss(outputs['sem1'], targets['sem1'])
        loss_sem2 = self.calc_sem_loss(outputs['sem2'], targets['sem2'])
        
        loss_aux = 0.0
        # Deep Supervision logic
        if 'sem1_aux' in outputs and outputs['sem1_aux'] is not None:
            for aux_logits in outputs['sem1_aux']:
                aux_logits = F.interpolate(aux_logits, size=targets['sem1'].shape[-2:], mode='bilinear', align_corners=False)
                loss_aux += 0.4 * self.calc_sem_loss(aux_logits, targets['sem1'])
                
        if 'sem2_aux' in outputs and outputs['sem2_aux'] is not None:
            for aux_logits in outputs['sem2_aux']:
                aux_logits = F.interpolate(aux_logits, size=targets['sem2'].shape[-2:], mode='bilinear', align_corners=False)
                loss_aux += 0.4 * self.calc_sem_loss(aux_logits, targets['sem2'])

        loss_sem = (loss_sem1 + loss_sem2) / 2 + loss_aux

        if self.bcd_bce.pos_weight.device != outputs['bcd'].device:
            self.bcd_bce.pos_weight = self.bcd_bce.pos_weight.to(outputs['bcd'].device)

        target_bcd = targets['bcd'].unsqueeze(1).float()
        loss_bcd = self.bcd_bce(outputs['bcd'], target_bcd) + self.bcd_dice(outputs['bcd'], target_bcd)
        
        loss_scl = self.scl_loss(outputs['sem1'], outputs['sem2'], targets['bcd'])
        loss_kld = outputs.get('kld_loss', torch.tensor(0.0, device=outputs['sem1'].device))
        
        loss_flow = torch.tensor(0.0, device=outputs['sem1'].device)
        if 'flow' in outputs and outputs['flow'] is not None:
            loss_flow = self.flow_loss(outputs['flow'])

        total_loss = loss_sem + loss_bcd + \
                     (self.lambda_scl * loss_scl) + \
                     (self.lambda_kld * loss_kld) + \
                     (self.lambda_flow * loss_flow)
                     
        return {
            "total": total_loss, "sem": loss_sem, "bcd": loss_bcd,
            "scl": loss_scl, "kld": loss_kld, "flow": loss_flow
        }