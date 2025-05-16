import os
import argparse
import torch
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import trimesh

# Import des modules personnalisés
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hunyuan3d_adapted import Hunyuan3DGlassesAdapter
from utils.dataset import create_dataloader

def visualize_results(image, mesh, output_path):
    """Visualisation des résultats (image d'entrée et rendu du maillage)"""
    # Conversion de l'image tensor en numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
        # Dénormalisation
        image = (image * 0.5 + 0.5).clip(0, 1)
    
    # Rendu du maillage
    renderer = trimesh.Scene(mesh)
    rendered = renderer.save_image(resolution=(512, 512), visible=True)
    rendered = np.array(Image.open(rendered))
    
    # Création de la figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Affichage de l'image d'entrée
    axes[0].imshow(image)
    axes[0].set_title("Image d'entrée")
    axes[0].axis("off")
    
    # Affichage du rendu 3D
    axes[1].imshow(rendered)
    axes[1].set_title("Modèle 3D généré")
    axes[1].axis("off")
    
    # Sauvegarde de la figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_model(model, dataloader, args):
    """Évaluation du modèle sur un dataset"""
    device = next(model.parameters()).device
    model.eval()
    
    # Création des répertoires de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "metrics"), exist_ok=True)
    
    # Métriques globales
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Évaluation")):
            # Transfert des données sur le device
            images = batch["image"].to(device)
            image_names = batch["image_name"]
            
            # Génération des modèles 3D
            outputs = model(images)
            
            # Calcul des métriques
            metrics = model.compute_metrics(images, outputs)
            all_metrics.append(metrics)
            
            # Sauvegarde des métriques individuelles
            metrics_path = os.path.join(args.output_dir, "metrics", f"{os.path.splitext(image_names[0])[0]}_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Sauvegarde des maillages
            for i, image_name in enumerate(image_names):
                base_name = os.path.splitext(image_name)[0]
                
                # Sauvegarde du maillage
                mesh_path = os.path.join(args.output_dir, "meshes", f"{base_name}.glb")
                model.save_mesh(outputs[i], mesh_path)
                
                # Visualisation
                vis_path = os.path.join(args.output_dir, "visualizations", f"{base_name}.png")
                visualize_results(images[i], outputs[i], vis_path)
    
    # Calcul des métriques moyennes
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    # Sauvegarde des métriques moyennes
    avg_metrics_path = os.path.join(args.output_dir, "average_metrics.json")
    with open(avg_metrics_path, "w") as f:
        json.dump(avg_metrics, f, indent=2)
    
    print("Métriques moyennes:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return avg_metrics

def main(args):
    """Fonction principale d'évaluation"""
    # Configuration du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device: {device}")
    
    # Chargement du modèle
    model = Hunyuan3DGlassesAdapter().to(device)
    
    # Chargement des poids du modèle
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Modèle chargé depuis: {args.model_path}")
    
    # Création du dataloader
    test_loader = create_dataloader(
        args.test_data,
        batch_size=1,  # Batch size de 1 pour l'évaluation
        num_workers=args.num_workers,
        shuffle=False,
        augment=False
    )
    
    # Évaluation du modèle
    metrics = evaluate_model(model, test_loader, args)
    
    # Génération d'un rapport d'évaluation
    report = {
        "model_path": args.model_path,
        "test_data": args.test_data,
        "num_samples": len(test_loader),
        "metrics": metrics,
        "output_dir": args.output_dir
    }
    
    # Sauvegarde du rapport
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Évaluation terminée. Rapport sauvegardé dans: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation du modèle Hunyuan3D pour les lunettes")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le modèle à évaluer")
    parser.add_argument("--test_data", type=str, required=True, help="Répertoire des données de test")
    parser.add_argument("--output_dir", type=str, default="outputs/generated_samples", help="Répertoire de sortie")
    parser.add_argument("--num_workers", type=int, default=4, help="Nombre de workers pour le dataloader")
    
    args = parser.parse_args()
    main(args)
