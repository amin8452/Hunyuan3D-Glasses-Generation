#!/usr/bin/env python
"""
Script for fine-tuning the Hunyuan3D 2.0 model for glasses generation.
This script implements:
1. Loading the pre-trained Hunyuan3D 2.0 model
2. Freezing/unfreezing specific layers
3. Fine-tuning on glasses dataset
4. Saving checkpoints and monitoring progress
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hunyuan3d_adapted import Hunyuan3DGlassesAdapter
from utils.dataset import create_dataloader

def freeze_layers(model, freeze_pattern=None):
    """
    Freeze specific layers of the model based on a pattern.
    
    Args:
        model: The model to freeze layers in
        freeze_pattern: String pattern to match layer names to freeze
                        If None, no layers are frozen
    """
    if freeze_pattern is None:
        return
    
    for name, param in model.named_parameters():
        if freeze_pattern in name:
            param.requires_grad = False
            print(f"Freezing layer: {name}")
        else:
            param.requires_grad = True

def unfreeze_layers(model, unfreeze_pattern=None):
    """
    Unfreeze specific layers of the model based on a pattern.
    
    Args:
        model: The model to unfreeze layers in
        unfreeze_pattern: String pattern to match layer names to unfreeze
                          If None, all layers are unfrozen
    """
    if unfreeze_pattern is None:
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
        return
    
    for name, param in model.named_parameters():
        if unfreeze_pattern in name:
            param.requires_grad = True
            print(f"Unfreezing layer: {name}")

def load_pretrained_model(model_path, device):
    """
    Load a pre-trained Hunyuan3D model.
    
    Args:
        model_path: Path to the pre-trained model
        device: Device to load the model on
    
    Returns:
        The loaded model
    """
    print(f"Loading pre-trained model from {model_path}...")
    
    # Initialize the model
    model = Hunyuan3DGlassesAdapter().to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    print("Pre-trained model loaded successfully")
    return model

def train_epoch(model, dataloader, optimizer, device, epoch, scaler=None, use_amp=False):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        images = batch["image"].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with or without AMP
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = outputs.loss
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Update progress bar
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / (batch_idx + 1)})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    metrics = {"FID": 0, "LPIPS": 0, "SSIM": 0, "PSNR": 0, "Symmetry": 0, "Wearability": 0}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["image"].to(device)
            outputs = model(images)
            
            # Calculate metrics
            batch_metrics = model.compute_metrics(images, outputs)
            
            # Accumulate metrics
            for k in metrics:
                metrics[k] += batch_metrics[k] / len(dataloader)
    
    return metrics

def fine_tune(args):
    """Main fine-tuning function"""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pre-trained model
    model = load_pretrained_model(args.pretrained_model, device)
    
    # Freeze/unfreeze layers based on strategy
    if args.fine_tuning_strategy == "full":
        # Fine-tune the entire model
        unfreeze_layers(model)
    elif args.fine_tuning_strategy == "last_layers":
        # Freeze the base model, fine-tune only the adaptation layers
        freeze_layers(model, "pipeline")
        unfreeze_layers(model, "eye_shape_adapter")
        unfreeze_layers(model, "texture_projection")
        unfreeze_layers(model, "geometry_projection")
    elif args.fine_tuning_strategy == "progressive":
        # Start with only adaptation layers, then gradually unfreeze more
        freeze_layers(model, "pipeline")
        unfreeze_layers(model, "eye_shape_adapter")
        unfreeze_layers(model, "texture_projection")
        unfreeze_layers(model, "geometry_projection")
        # We'll unfreeze more layers during training
    
    # Create dataloaders
    train_loader = create_dataloader(
        args.train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        augment=True
    )
    
    val_loader = create_dataloader(
        args.val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        augment=False
    )
    
    # Set up optimizer
    if args.optimizer == "adamw":
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # Set up scheduler
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=args.epochs * len(train_loader)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")
    
    # Set up AMP
    scaler = GradScaler() if args.use_amp else None
    
    # Set up TensorBoard
    log_dir = os.path.join(args.output_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    best_metric = float("inf")  # Lower is better for FID
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Progressive unfreezing (if enabled)
        if args.fine_tuning_strategy == "progressive" and epoch == args.progressive_unfreeze_epoch:
            print("Progressive unfreezing: Unfreezing more layers...")
            unfreeze_layers(model, "pipeline.unet")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch,
            scaler=scaler, use_amp=args.use_amp
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Update scheduler
        if args.scheduler == "cosine":
            scheduler.step()
        # OneCycleLR is updated after each batch in train_epoch
        
        # Log metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"Metrics/{k}", v, epoch)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print("Validation Metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Save checkpoint
        is_best = val_metrics["FID"] < best_metric
        if is_best:
            best_metric = val_metrics["FID"]
            print(f"New best model! FID: {best_metric:.4f}")
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": val_metrics,
            "best_metric": best_metric
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_model_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, f"metrics_epoch_{epoch}.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_metrics": val_metrics,
                "is_best": is_best
            }, f, indent=2)
    
    print(f"Fine-tuning complete! Best FID: {best_metric:.4f}")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Hunyuan3D for glasses generation")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    
    # Model arguments
    parser.add_argument("--pretrained_model", type=str, required=True, help="Path to pre-trained model")
    parser.add_argument("--fine_tuning_strategy", type=str, default="last_layers", 
                        choices=["full", "last_layers", "progressive"],
                        help="Fine-tuning strategy")
    parser.add_argument("--progressive_unfreeze_epoch", type=int, default=10,
                        help="Epoch to unfreeze more layers in progressive strategy")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw"],
                        help="Optimizer to use")
    parser.add_argument("--scheduler", type=str, default="cosine", 
                        choices=["cosine", "onecycle"],
                        help="Learning rate scheduler")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="models/fine_tuned",
                        help="Output directory for fine-tuned models")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    
    args = parser.parse_args()
    fine_tune(args)
