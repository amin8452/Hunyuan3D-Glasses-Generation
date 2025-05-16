import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import trimesh

# Import des modules personnalisés
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hunyuan3d_adapted import GlassesGenerator

def visualize_3d_model(model_path, output_path=None, show=True):
    """Visualisation d'un modèle 3D avec différentes vues"""
    # Chargement du modèle
    mesh = trimesh.load(model_path)
    
    # Création de la scène
    scene = trimesh.Scene(mesh)
    
    # Génération de 4 vues différentes
    angles = [0, 90, 180, 270]
    renders = []
    
    for angle in angles:
        # Rotation de la scène
        rotated_scene = scene.copy()
        rotated_scene.camera_transform = trimesh.transformations.rotation_matrix(
            angle * np.pi / 180, [0, 1, 0]
        )
        
        # Rendu
        render = rotated_scene.save_image(resolution=(512, 512), visible=True)
        renders.append(np.array(Image.open(render)))
    
    # Création de la figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, (render, angle) in enumerate(zip(renders, angles)):
        axes[i].imshow(render)
        axes[i].set_title(f"Vue {angle}°")
        axes[i].axis("off")
    
    plt.tight_layout()
    
    # Sauvegarde ou affichage
    if output_path:
        plt.savefig(output_path)
        print(f"Visualisation sauvegardée dans: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def main(args):
    """Fonction principale pour la génération de lunettes 3D"""
    # Vérification de l'existence du fichier d'entrée
    if not os.path.exists(args.input_image):
        print(f"Erreur: Le fichier d'entrée {args.input_image} n'existe pas.")
        return
    
    # Création du répertoire de sortie
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    
    # Chargement du générateur
    generator = GlassesGenerator.load_from_checkpoint(args.model_path)
    print(f"Modèle chargé depuis: {args.model_path}")
    
    # Génération du modèle 3D
    print(f"Génération du modèle 3D à partir de: {args.input_image}")
    output_path = generator.generate_glasses(args.input_image, args.output_model)
    print(f"Modèle 3D généré: {output_path}")
    
    # Visualisation du modèle 3D
    if args.visualize:
        vis_path = os.path.splitext(args.output_model)[0] + "_visualization.png"
        visualize_3d_model(args.output_model, vis_path, show=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génération de lunettes 3D à partir d'une image")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le modèle entraîné")
    parser.add_argument("--input_image", type=str, required=True, help="Chemin vers l'image d'entrée")
    parser.add_argument("--output_model", type=str, required=True, help="Chemin pour le modèle 3D généré")
    parser.add_argument("--visualize", action="store_true", help="Générer une visualisation du modèle 3D")
    
    args = parser.parse_args()
    main(args)
