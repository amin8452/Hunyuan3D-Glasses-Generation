import os
import argparse
import json
import shutil
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import glob

def create_directory(directory):
    """Crée un répertoire s'il n'existe pas déjà"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_image(image_path, output_dir, target_size=(512, 512)):
    """Prétraite une image et la sauvegarde dans le répertoire de sortie"""
    # Chargement de l'image
    image = Image.open(image_path).convert("RGB")
    
    # Redimensionnement
    image = image.resize(target_size, Image.LANCZOS)
    
    # Extraction du nom de fichier
    filename = os.path.basename(image_path)
    base_name, ext = os.path.splitext(filename)
    
    # Sauvegarde de l'image prétraitée
    output_path = os.path.join(output_dir, "images", f"{base_name}.png")
    image.save(output_path)
    
    # Détection des contours pour les métadonnées
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Détection des contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calcul des caractéristiques des lunettes
    if contours:
        # Trouver le plus grand contour (supposé être les lunettes)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calcul du rectangle englobant
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calcul des moments pour trouver le centre
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        # Calcul de l'orientation
        if len(largest_contour) >= 5:
            (x_ellipse, y_ellipse), (MA, ma), angle = cv2.fitEllipse(largest_contour)
        else:
            angle = 0
        
        # Métadonnées
        metadata = {
            "width": w,
            "height": h,
            "center_x": cx,
            "center_y": cy,
            "angle": angle,
            "area": cv2.contourArea(largest_contour)
        }
    else:
        # Métadonnées par défaut si aucun contour n'est trouvé
        metadata = {
            "width": target_size[0],
            "height": target_size[1],
            "center_x": target_size[0] // 2,
            "center_y": target_size[1] // 2,
            "angle": 0,
            "area": 0
        }
    
    return f"{base_name}.png", metadata

def main(args):
    """Fonction principale pour le prétraitement des images"""
    # Création des répertoires de sortie
    output_images_dir = os.path.join(args.output_dir, "images")
    create_directory(output_images_dir)
    
    # Recherche des images dans le répertoire d'entrée
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
    
    print(f"Trouvé {len(image_paths)} images à prétraiter")
    
    # Prétraitement des images
    metadata = {}
    for image_path in tqdm(image_paths, desc="Prétraitement des images"):
        filename, image_metadata = preprocess_image(
            image_path, 
            args.output_dir, 
            target_size=(args.resolution, args.resolution)
        )
        metadata[filename] = image_metadata
    
    # Sauvegarde des métadonnées
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Prétraitement terminé. {len(metadata)} images traitées.")
    print(f"Métadonnées sauvegardées dans {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prétraitement des images de lunettes")
    parser.add_argument("--input_dir", type=str, required=True, help="Répertoire contenant les images d'entrée")
    parser.add_argument("--output_dir", type=str, required=True, help="Répertoire de sortie pour les images prétraitées")
    parser.add_argument("--resolution", type=int, default=512, help="Résolution cible des images (carré)")
    
    args = parser.parse_args()
    main(args)
