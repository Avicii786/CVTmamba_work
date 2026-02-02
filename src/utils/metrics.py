import numpy as np
import math

class SCDMetrics:
    """
    Computes Semantic Change Detection metrics based on the SECOND dataset evaluation protocol.
    
    Metrics calculated:
    
    1. BCD Metrics (Binary Change Detection):
       - Rec (Recall): TP / (TP + FN)
       - Pre (Precision): TP / (TP + FP)
       - OA_bcd (Overall Accuracy): (TP + TN) / Total
       - F1_bcd (F1 Score): 2 * (Pre * Rec) / (Pre + Rec)
       - IoU_bcd (Intersection over Union): TP / (TP + FP + FN)
       - KC_bcd (Kappa Coefficient): Binary Kappa
       
    2. SCD Metrics (Semantic Change Detection):
       - OA (Overall Accuracy): Multiclass Pixel Accuracy
       - mIoU (Mean IoU): Average Intersection over Union for semantic classes
       - mAcc (Mean Accuracy): Average Pixel Accuracy (Recall) per class
       - F1_scd: Often defined as the Binary F1 (same as F1_bcd) in literature like SECOND.
       - SeK (Separated Kappa): Kappa_no_change * exp(IoU_change)
    """
    def __init__(self, num_classes=7):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.count = 0

    def _fast_hist(self, label, pred):
        """
        Standard confusion matrix calculation.
        Rows: True Label
        Cols: Prediction
        """
        mask = (label >= 0) & (label < self.num_classes)
        hist = np.bincount(
            self.num_classes * label[mask].astype(int) + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, preds, labels):
        """
        Updates the internal confusion matrix.
        preds: [B, H, W] or [B, C, H, W] (numpy or tensor)
        labels: [B, H, W] (numpy or tensor)
        """
        # Convert to numpy if tensor
        if hasattr(preds, 'cpu'): preds = preds.cpu().numpy()
        if hasattr(labels, 'cpu'): labels = labels.cpu().numpy()
        
        # If preds are logits/probabilities [B, C, H, W], take argmax
        if preds.ndim == 4:
            preds = np.argmax(preds, axis=1)
            
        # Flatten
        preds = preds.flatten()
        labels = labels.flatten()
        
        self.hist += self._fast_hist(labels, preds)
        self.count += 1

    def calculate_kappa(self, hist):
        """Calculates Cohen's Kappa for a given confusion matrix."""
        if hist.sum() == 0:
            return 0
        
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        
        if pe == 1:
            return 0
        return (po - pe) / (1 - pe)

    def get_results(self):
        """
        Returns a dictionary of calculated metrics.
        """
        hist = self.hist
        total_pixels = hist.sum() + 1e-10
        
        # --- 1. SCD Metrics (Multiclass) ---
        
        # OA (Overall Accuracy - Multiclass)
        oa = np.diag(hist).sum() / total_pixels
        
        # Per-Class Accuracy (Recall)
        # acc = diag / row_sum
        acc_per_class = np.diag(hist) / (hist.sum(axis=1) + 1e-10)
        mAcc = np.nanmean(acc_per_class)

        # Mean IoU (mIoU)
        # IoU per class = TP / (TP + FP + FN)
        # iu = diag / (row_sum + col_sum - diag)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
        mean_iou = np.nanmean(iu)
        
        # --- 2. BCD Metrics (Binary: Change vs No-Change) ---
        # Class 0 is 'Unchanged'. We aggregate all change classes (1 to N) into a single 'Change' class.
        # This reduces the NxN matrix to a 2x2 matrix: [[TN, FP], [FN, TP]]
        
        tn = hist[0, 0]
        fp = hist[0, 1:].sum() # Ground Truth 0 (No Change), Pred > 0 (Change)
        fn = hist[1:, 0].sum() # Ground Truth > 0 (Change), Pred 0 (No Change)
        tp = hist[1:, 1:].sum() # Ground Truth > 0 (Change), Pred > 0 (Change)
        
        # Construct Binary Confusion Matrix
        hist_bcd = np.array([[tn, fp], [fn, tp]])
        
        # BCD: Precision (Pre)
        pre_bcd = tp / (tp + fp + 1e-10)
        
        # BCD: Recall (Rec)
        rec_bcd = tp / (tp + fn + 1e-10)
        
        # BCD: F1 Score (F1)
        f1_bcd = 2 * (pre_bcd * rec_bcd) / (pre_bcd + rec_bcd + 1e-10)
        
        # BCD: Overall Accuracy (OA_bcd)
        oa_bcd = (tp + tn) / total_pixels
        
        # BCD: Intersection over Union (IoU_bcd)
        iou_bcd = tp / (tp + fp + fn + 1e-10)
        
        # BCD: Kappa Coefficient (KC_bcd)
        kc_bcd = self.calculate_kappa(hist_bcd)
        
        # --- 3. Advanced SCD Metrics (SeK) ---
        
        # Separated Kappa (SeK)
        # SeK = Kappa_no_change * e^(IoU_change)
        # Kappa_no_change is calculated on a confusion matrix where the (0,0) element is set to 0.
        
        hist_n0 = hist.copy()
        hist_n0[0, 0] = 0 # Ignore true negatives for SeK's kappa calculation
        kappa_n0 = self.calculate_kappa(hist_n0)
        
        sek = (kappa_n0 * math.exp(iou_bcd)) / math.e
        
        # 4. Combined Score (Often used for ranking)
        score = 0.3 * mean_iou + 0.7 * sek

        return {
            # SCD Metrics
            "OA": oa,               
            "mIoU": mean_iou,
            "mAcc": mAcc,           # Mean Accuracy (Recall) per class
            "SeK": sek,
            "F1_scd": f1_bcd,       # In SECOND paper context, F1_scd is the binary F1
            "iou_per_class": iu,
            "acc_per_class": acc_per_class, # Accuracy per class
            
            # BCD Metrics (Explicitly named as per request)
            "Rec": rec_bcd,         # Recall
            "Pre": pre_bcd,         # Precision
            "OA_bcd": oa_bcd,       # Binary OA
            "F1_bcd": f1_bcd,       # Binary F1
            "IoU_bcd": iou_bcd,     # Binary IoU (Foreground IoU)
            "KC_bcd": kc_bcd,       # Binary Kappa
            
            # Composite & Debug
            "score": score,
            "hist": hist,
            "c2_hist": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
        }