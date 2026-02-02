import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import your modules
from src.datasets.dataset import SCDDataset
from src.models.lambda_scd import LambdaSCD
from src.utils.losses import LambdaSCDLoss
from src.utils.metrics import SCDMetrics

def get_args():
    parser = argparse.ArgumentParser(description='Train LambdaSCD')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--dataset_name', type=str, default='SECOND', choices=['SECOND', 'LandsatSCD'], help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps to simulate larger batches')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./runs', help='Directory for TensorBoard logs')
    return parser.parse_args()

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    running_bcd_loss = 0.0
    running_sem_loss = 0.0
    
    optimizer.zero_grad() # Initialize gradients
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for i, batch in enumerate(pbar):
        # [FIX] Explicitly cast images to float32 to prevent DoubleTensor errors
        img_A = batch['img_A'].to(device).float()
        img_B = batch['img_B'].to(device).float()
        
        label_A = batch['sem1'].to(device)
        label_B = batch['sem2'].to(device)
        target_bcd = batch['bcd'].to(device)

        # Forward
        outputs = model(img_A, img_B)
        
        targets = {
            "sem1": label_A,
            "sem2": label_B,
            "bcd": target_bcd
        }
        
        # Loss
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["total"]
        
        # Normalize loss for accumulation
        loss = loss / accumulation_steps
        
        # Backward (Accumulate gradients)
        loss.backward()
        
        # Step Optimizer every 'accumulation_steps'
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
        
        # Logging (multiply back to show real loss scale)
        current_loss = loss.item() * accumulation_steps
        running_loss += current_loss
        running_bcd_loss += loss_dict['bcd'].item()
        running_sem_loss += loss_dict['sem'].item()
        
        pbar.set_postfix({
            "Loss": f"{current_loss:.4f}", 
            "BCD": f"{loss_dict['bcd'].item():.4f}"
        })
        
        # Step-level logging
        global_step = (epoch - 1) * len(loader) + i
        if i % 10 == 0:
            writer.add_scalar('Train/Step_Loss', current_loss, global_step)
            for name, param in model.named_parameters():
                if "alpha" in name:
                    writer.add_scalar(f"Params/{name}", torch.tanh(param).item(), global_step)

    # Epoch-level logging
    avg_loss = running_loss / len(loader)
    avg_bcd = running_bcd_loss / len(loader)
    avg_sem = running_sem_loss / len(loader)
    
    writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
    writer.add_scalar('Train/Epoch_BCD_Loss', avg_bcd, epoch)
    writer.add_scalar('Train/Epoch_Sem_Loss', avg_sem, epoch)
    
    return avg_loss

def validate(model, loader, criterion, metrics, device, epoch, writer):
    """
    Runs validation and calculates both Loss and Metrics.
    """
    model.eval()
    running_loss = 0.0
    metrics.reset() # Reset confusion matrix
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Val]"):
            # [FIX] Explicitly cast to float32
            img_A = batch['img_A'].to(device).float()
            img_B = batch['img_B'].to(device).float()
            label_A = batch['sem1'].to(device)
            label_B = batch['sem2'].to(device)
            target_bcd = batch['bcd'].to(device)
            
            outputs = model(img_A, img_B)
            
            # 1. Calculate Loss
            targets = {
                "sem1": label_A,
                "sem2": label_B,
                "bcd": target_bcd
            }
            loss_dict = criterion(outputs, targets)
            running_loss += loss_dict["total"].item()
            
            # 2. Update Metrics
            pred_sem1 = torch.argmax(outputs['sem1'], dim=1)
            pred_sem2 = torch.argmax(outputs['sem2'], dim=1)
            
            metrics.update(pred_sem1, label_A)
            metrics.update(pred_sem2, label_B)
            
    avg_loss = running_loss / len(loader)
    results = metrics.get_results()
    
    # TensorBoard Logging
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/SCD_Score', results['score'] * 100, epoch)
    writer.add_scalar('Val/mIoU', results['mIoU'] * 100, epoch)
    writer.add_scalar('Val/SeK', results['SeK'] * 100, epoch)
    writer.add_scalar('Val/F1_scd', results['F1_scd'] * 100, epoch)
    writer.add_scalar('Val/OA', results['OA'] * 100, epoch)
    writer.add_scalar('Val/BCD_F1', results['F1_bcd'] * 100, epoch)
    writer.add_scalar('Val/BCD_IoU', results['IoU_bcd'] * 100, epoch)
    writer.add_scalar('Val/BCD_OA', results['OA_bcd'] * 100, epoch)
    
    return avg_loss, results

def main():
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"TensorBoard logging to: {args.log_dir}")
    
    # --- Datasets ---
    print(f"Initializing {args.dataset_name} Datasets...")
    
    train_ds = SCDDataset(
        root=args.data_root, 
        mode='train', 
        dataset_name=args.dataset_name,
        random_flip=True, 
        random_swap=True
    )
    
    val_ds = SCDDataset(
        root=args.data_root, 
        mode='val', 
        dataset_name=args.dataset_name,
        random_flip=False,
        random_swap=False
    )
    
    num_classes = train_ds.num_classes
    print(f"Detected {num_classes} classes for {args.dataset_name}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # --- Model ---
    print("Building Model...")
    model = LambdaSCD(num_classes=num_classes).to(device)
    
    # --- Optimization ---
    criterion = LambdaSCDLoss(ignore_index=255, lambda_scl=0.5, lambda_kld=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    val_metrics = SCDMetrics(num_classes=num_classes)
    best_score = 0.0 
    
    print("Starting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        # 1. Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            accumulation_steps=args.accumulation_steps
        )
        
        # 2. Validate
        val_loss, val_results = validate(model, val_loader, criterion, val_metrics, device, epoch, writer)
        
        # 3. Log
        score = val_results['score'] * 100
        mIoU = val_results['mIoU'] * 100
        sek = val_results['SeK'] * 100
        f1_scd = val_results['F1_scd'] * 100
        f1_bcd = val_results['F1_bcd'] * 100
        
        print(f"Epoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val Score: {score:.2f} | mIoU: {mIoU:.2f} | SeK: {sek:.2f} | F1-SCD: {f1_scd:.2f} | F1-BCD: {f1_bcd:.2f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LR', current_lr, epoch)
        
        scheduler.step()
        
        # 4. Save Best
        if score > best_score:
            best_score = score
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New Best Model Saved! (Score: {score:.2f})")
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'score': score
        }, os.path.join(args.save_dir, 'last_checkpoint.pth'))

    writer.close()

if __name__ == "__main__":
    main()