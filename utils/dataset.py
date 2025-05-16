import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import glob

class GlassesDataset(Dataset):
    """Dataset pour les images de lunettes et leurs modèles 3D correspondants"""
    
    def __init__(self, data_dir, transform=None, target_size=(512, 512), include_3d=False):
        """
        Initialisation du dataset
        
        Args:
            data_dir (str): Chemin vers le répertoire de données
            transform (callable, optional): Transformations à appliquer aux images
            target_size (tuple): Taille cible des images
            include_3d (bool): Si True, charge également les modèles 3D correspondants
        """
        self.data_dir = data_dir
        self.include_3d = include_3d
        
        # Liste des fichiers d'images
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.jpg")))
        self.image_paths.extend(sorted(glob.glob(os.path.join(data_dir, "images", "*.png"))))
        
        # Transformations par défaut si aucune n'est fournie
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
        
        # Chargement des métadonnées si disponibles
        metadata_path = os.path.join(data_dir, "metadata.json")
        self.metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Récupération d'un élément du dataset"""
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        
        # Chargement et transformation de l'image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        
        # Préparation du dictionnaire de retour
        sample = {
            "image": image_tensor,
            "image_path": image_path,
            "image_name": image_name
        }
        
        # Ajout des métadonnées si disponibles
        if image_name in self.metadata:
            sample["metadata"] = self.metadata[image_name]
        
        # Chargement du modèle 3D si demandé
        if self.include_3d:
            # Détermination du chemin du modèle 3D
            model_name = os.path.splitext(image_name)[0] + ".glb"
            model_path = os.path.join(self.data_dir, "models", model_name)
            
            if os.path.exists(model_path):
                sample["model_path"] = model_path
            else:
                sample["model_path"] = None
        
        return sample

class GlassesAugmentedDataset(GlassesDataset):
    """Extension du dataset avec des augmentations spécifiques aux lunettes"""
    
    def __init__(self, data_dir, transform=None, target_size=(512, 512), include_3d=False):
        super().__init__(data_dir, transform, target_size, include_3d)
        
        # Augmentations spécifiques aux lunettes
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ])
    
    def __getitem__(self, idx):
        """Récupération d'un élément du dataset avec augmentation"""
        sample = super().__getitem__(idx)
        
        # Application des augmentations
        image = transforms.ToPILImage()(sample["image"])
        augmented_image = self.augmentations(image)
        sample["image"] = self.transform(augmented_image)
        
        return sample

def create_dataloader(data_dir, batch_size=8, num_workers=4, shuffle=True, augment=True, include_3d=False):
    """Création d'un DataLoader pour le dataset de lunettes"""
    if augment:
        dataset = GlassesAugmentedDataset(data_dir, include_3d=include_3d)
    else:
        dataset = GlassesDataset(data_dir, include_3d=include_3d)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
