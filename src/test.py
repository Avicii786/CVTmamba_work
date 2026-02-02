import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import modules
from src.datasets.dataset import SCDDataset
from src.models.lambda_scd import LambdaSCD
from src.utils.metrics import SCDMetrics

def get_args():
    parser = argparse.ArgumentParser(description='Test LambdaSCD')

    parser.add_argument('--data_root', 
                        type=str, 
                        required=True, 
                        help='Path to dataset root')
    
    parser.add_argument('--dataset_name', 
                        type=str, 
                        default='SECOND', 
                        choices=['SECOND', 'LandsatSCD'], 
                        help='Dataset name')
    
    parser.add_argument('--checkpoint', 
                        type=str, 
                        required=True, 
                        help='Path to .pth checkpoint file')
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=4, 
                        help='Batch size')
    
    parser.add_argument('--num_workers', 
                        type=int, 
                        default=4, 
                        help='Dataloader workers')
    # num_classes is determined automatically by the dataset class
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}")

    # 1. Dataset
    print(f"Loading {args.dataset_name} Test Split...")
    # Changed mode from 'val' to 'test'
    test_ds = SCDDataset(
        root=args.data_root, 
        mode='test', 
        dataset_name=args.dataset_name,
        random_flip=False,
        random_swap=False
    )
    test_loader = DataLoader(test_ds, 
                             batch_size=args.batch_size, 
                             shuffle=False, 
                             num_workers=args.num_workers)
    
    num_classes = test_ds.num_classes
    print(f"Detected {num_classes} classes.")

    # 2. Model
    print(f"Loading Model from {args.checkpoint}...")
    model = LambdaSCD(num_classes=num_classes).to(device)
    
    # Load Weights
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # Support loading raw state dict
    
    model.eval()

    # --- INSPECT LEARNABLE PARAMETERS (Novelty Check) ---
    print("\n[Analysis] Checking Cross-Temporal Mixing Parameters (Alpha):")
    alpha_values = []
    for name, param in model.named_parameters():
        if "alpha" in name:
            eff_alpha = torch.tanh(param).item()
            alpha_values.append(eff_alpha)
            print(f"  {name}: raw={param.item():.4f} | effective_mixing={eff_alpha:.4f}")
    
    if alpha_values:
        avg_alpha = sum(alpha_values) / len(alpha_values)
        print(f"  > Average Effective Mixing Rate: {avg_alpha:.4f}")
    else:
        print("  (No alpha parameters found - check backbone version)")
    print("-" * 30 + "\n")

    # 3. Metrics
    metrics = SCDMetrics(num_classes=num_classes)

    print("Starting Inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Unpack dictionary batch
            img_A = batch['img_A'].to(device)
            img_B = batch['img_B'].to(device)
            # Labels can stay on CPU for metric calculation to save GPU memory/transfer
            label_A = batch['sem1'] 
            label_B = batch['sem2']
            
            # Forward pass
            outputs = model(img_A, img_B)
            
            # Get Semantic Predictions
            # The model outputs logits [B, C, H, W], we take argmax to get class indices
            pred_sem1 = torch.argmax(outputs['sem1'], dim=1).cpu().numpy()
            pred_sem2 = torch.argmax(outputs['sem2'], dim=1).cpu().numpy()
            
            # Get Targets
            target_sem1 = label_A.numpy()
            target_sem2 = label_B.numpy()
            
            # Update Metrics (Evaluate both T1 and T2 predictions for semantic accuracy)
            metrics.update(pred_sem1, target_sem1)
            metrics.update(pred_sem2, target_sem2)

    # 4. Print Results
    results = metrics.get_results()
    
    print("\n" + "="*40)
    print("   TEST RESULTS")
    print("="*40)
    
    print("--- SCD Metrics (Semantic) ---")
    print(f"OA (Overall Accuracy): {results['OA']*100:.2f}%")
    print(f"mIoU (Mean IoU):       {results['mIoU']*100:.2f}%")
    print(f"mAcc (Mean Recall):    {results['mAcc']*100:.2f}%")
    print(f"SeK (Separated Kappa): {results['SeK']*100:.2f}%")
    print(f"F1-SCD:                {results['F1_scd']*100:.2f}%")
    print(f"Composite Score:       {results['score']*100:.2f}")
    
    print("\n--- BCD Metrics (Binary Change) ---")
    print(f"Precision:             {results['Pre']*100:.2f}%")
    print(f"Recall:                {results['Rec']*100:.2f}%")
    print(f"F1 Score:              {results['F1_bcd']*100:.2f}%")
    print(f"IoU:                   {results['IoU_bcd']*100:.2f}%")
    print(f"OA (Binary):           {results['OA_bcd']*100:.2f}%")
    print(f"Kappa (Binary):        {results['KC_bcd']*100:.2f}")
    
    print("-" * 40)
    c2 = results['c2_hist']
    print(f"Binary CM: TP={c2['TP']}, FN={c2['FN']}, FP={c2['FP']}, TN={c2['TN']}")
    print("="*40)

if __name__ == "__main__":
    main()