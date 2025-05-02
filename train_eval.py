import os
import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import logging
import time
import multiprocessing
from ultralytics import YOLO
from kitti_data_loader import KITTI3DDataset
from model_components import (
    KITTIStereoDepthEstimator, 
    ModifiedStereoBoundingBoxRegressor, 
    Box3DRegressionLoss, 
    box_iou, 
    calculate_3d_iou
)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None
    
    features = []
    targets = []
    for feat_batch, targ_batch in batch:
        features.append(feat_batch)
        targets.append(targ_batch)
    features = torch.cat(features, dim=0)
    targets = torch.cat(targets, dim=0)
    
    return features, targets

def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-4, model_save_path='weights'):
    
    os.makedirs(model_save_path, exist_ok=True)
    
    criterion = Box3DRegressionLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': []
    }
    
    best_val_iou = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0
        start_time = time.time()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            features, targets = batch
            
            if features is None:
                continue
            
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            batch_size = features.size(0)
            train_loss += loss.item() * batch_size
            train_samples += batch_size
            
            pbar.set_postfix({'loss': train_loss / max(1, train_samples)})
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / max(1, train_samples)
        history['train_loss'].append(avg_train_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, "
                  f"Train Loss: {avg_train_loss:.4f}")
        
        model.eval()
        val_loss = 0.0
        val_samples = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for features, targets in pbar:
                if features is None:
                    continue
                
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                batch_size = features.size(0)
                val_loss += loss.item() * batch_size
                val_samples += batch_size
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                pbar.set_postfix({'loss': val_loss / max(1, val_samples)})
        
        avg_val_loss = val_loss / max(1, val_samples)
        history['val_loss'].append(avg_val_loss)
        
        if all_preds and all_targets:
            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)
            ious = []
            for i in range(len(all_preds)):
                iou = calculate_3d_iou(all_preds[i], all_targets[i])
                ious.append(iou)
            
            avg_iou = np.mean(ious)
            history['val_iou'].append(avg_iou)
            
            logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, Val 3D IoU: {avg_iou:.4f}')
            
            if avg_iou > best_val_iou:
                best_val_iou = avg_iou
                save_path = os.path.join(model_save_path, f'best_model_iou_{avg_iou:.4f}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_iou': avg_iou,
                }, save_path)
                logger.info(f'Model saved to {save_path}')
        else:
            logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, No validation predictions available')
        
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(model_save_path, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            logger.info(f'Checkpoint saved to {checkpoint_path}')
        scheduler.step(avg_val_loss)
    final_path = os.path.join(model_save_path, 'final_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    logger.info(f'Final model saved to {final_path}')
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Validation 3D IoU')
    plt.xlabel('Epoch')
    plt.ylabel('3D IoU')
    plt.title('Validation 3D IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'training_history.png'))
    plt.close()
    
    return model, history

def evaluate_model(model, test_loader):
    
    model.eval()
    test_loss = 0.0
    test_samples = 0
    all_preds = []
    all_targets = []
    
    criterion = Box3DRegressionLoss()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for features, targets in pbar:
            if features is None:
                continue
            
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            batch_size = features.size(0)
            test_loss += loss.item() * batch_size
            test_samples += batch_size
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    avg_test_loss = test_loss / max(1, test_samples)
    logger.info(f'Test Loss: {avg_test_loss:.4f}')
    
    if all_preds and all_targets:
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        iou_thresholds = [0.25, 0.5, 0.7]
        results = {}
        
        for threshold in iou_thresholds:
            above_threshold = 0
            for i in range(len(all_preds)):
                iou = calculate_3d_iou(all_preds[i], all_targets[i])
                if iou >= threshold:
                    above_threshold += 1
            
            accuracy = above_threshold / len(all_preds) if len(all_preds) > 0 else 0
            results[f'AP@{threshold}'] = accuracy
            logger.info(f'AP@{threshold}: {accuracy:.4f}')
        
        ious = []
        for i in range(len(all_preds)):
            iou = calculate_3d_iou(all_preds[i], all_targets[i])
            ious.append(iou)
        
        avg_iou = np.mean(ious)
        results['avg_iou'] = avg_iou
        logger.info(f'Average 3D IoU: {avg_iou:.4f}')
        
        pos_error = np.mean(np.sqrt(np.sum((all_preds[:, :3] - all_targets[:, :3]) ** 2, axis=1)))
        results['pos_error'] = pos_error
        logger.info(f'Average position error: {pos_error:.4f} meters')
        
        dim_error = np.mean(np.sqrt(np.sum((all_preds[:, 3:6] - all_targets[:, 3:6]) ** 2, axis=1)))
        results['dim_error'] = dim_error
        logger.info(f'Average dimension error: {dim_error:.4f} meters')
        
        rot_error = np.mean(np.abs(all_preds[:, 6] - all_targets[:, 6])) * 180 / np.pi
        results['rot_error'] = rot_error
        logger.info(f'Average rotation error: {rot_error:.4f} degrees')
        
        return results
    else:
        logger.warning("No valid predictions for evaluation")
        return {}

def main():
    
    parser = argparse.ArgumentParser(description='Train 3D Bounding Box Regressor on KITTI dataset')
    parser.add_argument('--kitti_root', required=True, help='Path to KITTI dataset root directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--output_dir', default='stereo_box_regressor_weights', help='Output directory for model weights')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only (no training)')
    parser.add_argument('--model_path', help='Path to trained model for evaluation')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers (set to 0 to avoid multiprocessing issues)')
    
    args = parser.parse_args()
    
    logger.info("Creating datasets")
    train_dataset = KITTI3DDataset(args.kitti_root, split='train')
    val_dataset = KITTI3DDataset(args.kitti_root, split='val')
    test_dataset = KITTI3DDataset(args.kitti_root, split='test')
    
    logger.info("Creating data loaders")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=args.workers,  
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=args.workers,  
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=args.workers,  
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    try:
        logger.info("Checking training data")
        sample_features, sample_targets = None, None
        for batch in train_loader:
            sample_features, sample_targets = batch
            if sample_features is not None:
                break
            
        if sample_features is not None:
            logger.info(f"Sample features shape: {sample_features.shape}")
            logger.info(f"Sample targets shape: {sample_targets.shape}")
            input_dim = sample_features.shape[1]
        else:
            logger.warning("No valid samples found in the training set, using default input dim")
            input_dim = 12288+5+4
    except Exception as e:
        logger.error(f"Error checking training data: {e}")
        logger.warning("Using default input dimension")
        input_dim = 12288+5+4
    
    model = ModifiedStereoBoundingBoxRegressor(input_dim=input_dim)
    model.to(device)
    logger.info(f"Initialized model with input dimension: {input_dim}")
    
    if args.eval_only:
        if not args.model_path:
            logger.error("pls specify --model_path")
            return
        
        logger.info(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info("Evaluating model on test set")
        test_results = evaluate_model(model, test_loader)
    else:
        logger.info("Starting training")
        model, history = train_model(
            model, 
            train_loader, 
            val_loader, 
            num_epochs=args.epochs, 
            lr=args.lr, 
            model_save_path=args.output_dir
        )
        logger.info("Evaluating model on test set")
        test_results = evaluate_model(model, test_loader)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()