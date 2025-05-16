#!/usr/bin/env python
"""
Script for hyperparameter optimization of the Hunyuan3D glasses generation model.
This script uses Optuna to find the best hyperparameters for training.
"""

import os
import argparse
import json
import torch
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import numpy as np
from datetime import datetime
import sys
from functools import partial

# Import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hunyuan3d_adapted import Hunyuan3DGlassesAdapter
from utils.dataset import create_dataloader
from scripts.fine_tuning import train_epoch, validate, load_pretrained_model

def objective(trial, args):
    """Optuna objective function for hyperparameter optimization"""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained model
    model = load_pretrained_model(args.pretrained_model, device)
    
    # Sample hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD", "RMSprop"])
    scheduler_name = trial.suggest_categorical("scheduler", ["cosine", "onecycle"])
    
    # Fine-tuning strategy
    fine_tuning_strategy = trial.suggest_categorical(
        "fine_tuning_strategy", 
        ["full", "last_layers", "progressive"]
    )
    
    # Apply fine-tuning strategy
    if fine_tuning_strategy == "full":
        # Fine-tune the entire model
        for param in model.parameters():
            param.requires_grad = True
    elif fine_tuning_strategy == "last_layers":
        # Freeze the base model, fine-tune only the adaptation layers
        for name, param in model.named_parameters():
            if "pipeline" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif fine_tuning_strategy == "progressive":
        # Start with only adaptation layers
        for name, param in model.named_parameters():
            if "pipeline" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    # Create dataloaders
    train_loader = create_dataloader(
        args.train_data,
        batch_size=batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        augment=True
    )
    
    val_loader = create_dataloader(
        args.val_data,
        batch_size=batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        augment=False
    )
    
    # Set up optimizer
    if optimizer_name == "AdamW":
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
        optimizer = SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name == "RMSprop":
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
        optimizer = RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    
    # Set up scheduler
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif scheduler_name == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=args.epochs * len(train_loader)
        )
    
    # Training loop
    best_metric = float("inf")  # Lower is better for FID
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Progressive unfreezing (if enabled)
        if fine_tuning_strategy == "progressive" and epoch == args.epochs // 3:
            # Unfreeze more layers halfway through training
            for name, param in model.named_parameters():
                if "pipeline.unet" in name:
                    param.requires_grad = True
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch,
            scaler=None, use_amp=False
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Update scheduler
        if scheduler_name == "cosine":
            scheduler.step()
        
        # Check if this is the best model
        current_metric = val_metrics["FID"]
        if current_metric < best_metric:
            best_metric = current_metric
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Report intermediate metric
        trial.report(current_metric, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_metric

def run_optimization(args):
    """Run the hyperparameter optimization"""
    # Create study name
    study_name = f"hunyuan3d_glasses_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create database URL
    db_path = os.path.join(args.output_dir, f"{study_name}.db")
    db_url = f"sqlite:///{db_path}"
    
    # Create sampler
    sampler = TPESampler(seed=args.seed)
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=db_url,
        direction="minimize",  # Minimize FID
        sampler=sampler,
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(
        partial(objective, args=args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=1  # Use 1 job to avoid GPU conflicts
    )
    
    # Print results
    print("Number of finished trials:", len(study.trials))
    
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value:", best_trial.value)
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save results
    results = {
        "best_params": best_trial.params,
        "best_value": best_trial.value,
        "n_trials": len(study.trials),
        "finished_trials": len([t for t in study.trials if t.state == TrialState.COMPLETE]),
        "pruned_trials": len([t for t in study.trials if t.state == TrialState.PRUNED]),
        "all_trials": [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value,
                "state": str(t.state)
            }
            for t in study.trials
        ]
    }
    
    results_path = os.path.join(args.output_dir, f"{study_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Train a model with the best hyperparameters
    if args.train_best:
        print("\nTraining model with best hyperparameters...")
        
        # Create command for fine-tuning script
        cmd = [
            "python", "scripts/fine_tuning.py",
            "--train_data", args.train_data,
            "--val_data", args.val_data,
            "--pretrained_model", args.pretrained_model,
            "--output_dir", os.path.join(args.output_dir, "best_model"),
            "--epochs", str(args.epochs),
            "--num_workers", str(args.num_workers)
        ]
        
        # Add best hyperparameters
        for key, value in best_trial.params.items():
            if key == "batch_size":
                cmd.extend(["--batch_size", str(value)])
            elif key == "lr":
                cmd.extend(["--lr", str(value)])
            elif key == "weight_decay":
                cmd.extend(["--weight_decay", str(value)])
            elif key == "optimizer":
                cmd.extend(["--optimizer", value.lower()])
            elif key == "scheduler":
                cmd.extend(["--scheduler", value])
            elif key == "fine_tuning_strategy":
                cmd.extend(["--fine_tuning_strategy", value])
            elif key == "momentum" and value > 0:
                # Note: momentum is not directly supported in fine_tuning.py
                print(f"Note: momentum={value} is not directly supported in fine_tuning.py")
        
        # Run the command
        import subprocess
        subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for Hunyuan3D glasses generation")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    
    # Model arguments
    parser.add_argument("--pretrained_model", type=str, required=True, help="Path to pre-trained model")
    
    # Optimization arguments
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs per trial")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="models/hyperopt",
                        help="Output directory for optimization results")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--train_best", action="store_true", 
                        help="Train a model with the best hyperparameters")
    
    args = parser.parse_args()
    run_optimization(args)
