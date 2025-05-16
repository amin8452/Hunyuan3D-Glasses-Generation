import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Import des modules personnalisés
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hunyuan3d_adapted import Hunyuan3DGlassesAdapter
from utils.dataset import create_dataloader

def train_epoch(model, dataloader, optimizer, device, epoch, scaler=None, use_amp=False):
    """Entraînement pour une époque"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        # Transfert des données sur le device
        images = batch["image"].to(device)
        
        # Mise à zéro des gradients
        optimizer.zero_grad()
        
        # Forward pass avec ou sans AMP
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = outputs.loss
            
            # Backward pass avec scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = outputs.loss
            
            # Backward pass standard
            loss.backward()
            optimizer.step()
        
        # Mise à jour de la barre de progression
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / (batch_idx + 1)})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    """Validation du modèle"""
    model.eval()
    metrics = {"FID": 0, "LPIPS": 0, "SSIM": 0, "PSNR": 0, "Symmetry": 0, "Wearability": 0}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["image"].to(device)
            outputs = model(images)
            
            # Calcul des métriques
            batch_metrics = model.compute_metrics(images, outputs)
            
            # Accumulation des métriques
            for k in metrics:
                metrics[k] += batch_metrics[k] / len(dataloader)
    
    return metrics

def save_checkpoint(model, optimizer, epoch, metrics, args, is_best=False):
    """Sauvegarde d'un checkpoint du modèle"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics
    }
    
    # Création du répertoire de sortie s'il n'existe pas
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Sauvegarde du checkpoint
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Si c'est le meilleur modèle, faire une copie
    if is_best:
        best_model_path = os.path.join(args.output_dir, "best_model.pt")
        torch.save(checkpoint, best_model_path)
        print(f"Meilleur modèle sauvegardé: {best_model_path}")
    
    return checkpoint_path

def main(args):
    """Fonction principale d'entraînement"""
    # Configuration du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device: {device}")
    
    # Création des dataloaders
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
    
    # Initialisation du modèle
    model = Hunyuan3DGlassesAdapter(texture_res=args.resolution).to(device)
    
    # Optimiseur
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Scaler pour AMP
    scaler = GradScaler() if args.use_amp else None
    
    # TensorBoard
    log_dir = os.path.join(args.output_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    
    # Variables de suivi
    best_metric = float("inf")  # Pour FID, plus bas est meilleur
    
    # Boucle d'entraînement
    for epoch in range(args.epochs):
        # Entraînement
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, 
            scaler=scaler, use_amp=args.use_amp
        )
        
        # Validation
        val_metrics = validate(model, val_loader, device)
        
        # Mise à jour du scheduler
        scheduler.step()
        
        # Logging
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val FID = {val_metrics['FID']:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"Metrics/{k}", v, epoch)
        
        # Sauvegarde du modèle
        is_best = val_metrics["FID"] < best_metric
        if is_best:
            best_metric = val_metrics["FID"]
        
        checkpoint_path = save_checkpoint(
            model, optimizer, epoch, val_metrics, args, is_best=is_best
        )
        
        # Sauvegarde des métriques
        metrics_path = os.path.join(args.output_dir, f"metrics_epoch_{epoch}.json")
        with open(metrics_path, "w") as f:
            json.dump(val_metrics, f, indent=2)
    
    print(f"Entraînement terminé. Meilleur FID: {best_metric:.4f}")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement du modèle Hunyuan3D pour les lunettes")
    parser.add_argument("--train_data", type=str, required=True, help="Répertoire des données d'entraînement")
    parser.add_argument("--val_data", type=str, required=True, help="Répertoire des données de validation")
    parser.add_argument("--batch_size", type=int, default=8, help="Taille du batch")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques")
    parser.add_argument("--lr", type=float, default=1e-4, help="Taux d'apprentissage")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--resolution", type=int, default=512, help="Résolution des textures")
    parser.add_argument("--num_workers", type=int, default=4, help="Nombre de workers pour le dataloader")
    parser.add_argument("--output_dir", type=str, default="models/saved", help="Répertoire de sortie")
    parser.add_argument("--use_amp", action="store_true", help="Utiliser Automatic Mixed Precision")
    
    args = parser.parse_args()
    main(args)
