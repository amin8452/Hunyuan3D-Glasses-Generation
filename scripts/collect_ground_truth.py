#!/usr/bin/env python
"""
Script to collect and process ground truth 3D models of glasses for evaluation.
This script:
1. Downloads 3D models from various sources
2. Processes and normalizes the models for consistent evaluation
3. Pairs 3D models with corresponding 2D images
4. Creates a ground truth dataset for evaluation
"""

import os
import argparse
import json
import requests
import zipfile
import shutil
import hashlib
import random
import numpy as np
import trimesh
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import concurrent.futures
import sys

# Add parent directory to path for importing custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sources for 3D glasses models
SOURCES = {
    "sketchfab": {
        "name": "Sketchfab",
        "api_key_env": "SKETCHFAB_API_KEY",
        "search_url": "https://api.sketchfab.com/v3/search",
        "model_url": "https://api.sketchfab.com/v3/models/{model_id}/download"
    },
    "turbosquid": {
        "name": "TurboSquid",
        "api_key_env": "TURBOSQUID_API_KEY",
        "search_url": "https://www.turbosquid.com/Search/Index.cfm",
        "model_url": None  # TurboSquid doesn't have a direct download API
    },
    "cgtrader": {
        "name": "CGTrader",
        "api_key_env": "CGTRADER_API_KEY",
        "search_url": "https://www.cgtrader.com/api/v1/search",
        "model_url": None  # CGTrader doesn't have a direct download API
    }
}

def create_directory_structure(base_dir):
    """Create the directory structure for the ground truth dataset"""
    # Main directories
    os.makedirs(os.path.join(base_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "processed"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "paired"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "renders"), exist_ok=True)
    
    print(f"Created directory structure in {base_dir}")

def download_sketchfab_model(model_id, api_key, output_dir):
    """Download a 3D model from Sketchfab"""
    if not api_key:
        print("Sketchfab API key not set. Skipping download.")
        return None
    
    headers = {"Authorization": f"Token {api_key}"}
    
    try:
        # Request download URL
        download_url_endpoint = SOURCES["sketchfab"]["model_url"].format(model_id=model_id)
        response = requests.get(download_url_endpoint, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("gltf", {}).get("url"):
            print(f"No download URL available for model {model_id}")
            return None
        
        download_url = data["gltf"]["url"]
        
        # Download the model
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Create a unique filename
        filename = f"sketchfab_{model_id}.zip"
        zip_path = os.path.join(output_dir, filename)
        
        # Save the zip file
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract the zip file
        extract_dir = os.path.join(output_dir, model_id)
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the .glb or .gltf file
        glb_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(extract_dir) 
                    for f in filenames if f.endswith('.glb')]
        gltf_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(extract_dir) 
                     for f in filenames if f.endswith('.gltf')]
        
        model_file = None
        if glb_files:
            model_file = glb_files[0]
        elif gltf_files:
            model_file = gltf_files[0]
        
        if model_file:
            # Copy to final location
            final_path = os.path.join(output_dir, f"sketchfab_{model_id}.glb")
            shutil.copy(model_file, final_path)
            
            # Clean up
            os.remove(zip_path)
            shutil.rmtree(extract_dir)
            
            return final_path
        else:
            print(f"No .glb or .gltf file found in the downloaded archive for model {model_id}")
            return None
        
    except Exception as e:
        print(f"Error downloading model {model_id}: {e}")
        return None

def search_sketchfab_models(query, api_key, max_models=50):
    """Search for 3D models on Sketchfab"""
    if not api_key:
        print("Sketchfab API key not set. Skipping search.")
        return []
    
    headers = {"Authorization": f"Token {api_key}"}
    models = []
    cursor = None
    
    with tqdm(total=max_models, desc="Searching Sketchfab") as pbar:
        while len(models) < max_models:
            params = {
                "q": query,
                "type": "models",
                "downloadable": True,
                "count": 24
            }
            
            if cursor:
                params["cursor"] = cursor
            
            try:
                response = requests.get(SOURCES["sketchfab"]["search_url"], headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data.get("results"):
                    break
                
                for model in data["results"]:
                    if len(models) >= max_models:
                        break
                    
                    # Check if model is downloadable and has appropriate license
                    if model.get("downloadable") and model.get("license", {}).get("slug") in ["cc-by", "cc-by-sa", "cc-by-nd", "cc-by-nc", "cc-by-nc-sa", "cc-by-nc-nd", "cc0"]:
                        models.append({
                            "id": model["uid"],
                            "name": model["name"],
                            "author": model["user"]["username"],
                            "thumbnail_url": model["thumbnails"]["images"][0]["url"],
                            "license": model["license"]["slug"],
                            "source": "sketchfab",
                            "page_url": f"https://sketchfab.com/models/{model['uid']}"
                        })
                        pbar.update(1)
                
                # Check if there are more results
                cursor = data.get("cursors", {}).get("next")
                if not cursor:
                    break
                
                time.sleep(0.5)  # Respect rate limits
                
            except Exception as e:
                print(f"Error searching Sketchfab: {e}")
                break
    
    print(f"Found {len(models)} models on Sketchfab")
    return models

def normalize_model(model_path, output_path):
    """Normalize a 3D model for consistent evaluation"""
    try:
        # Load the model
        mesh = trimesh.load(model_path)
        
        # Check if it's a scene or a mesh
        if isinstance(mesh, trimesh.Scene):
            # Extract the largest mesh from the scene
            geometries = list(mesh.geometry.values())
            if not geometries:
                print(f"No geometries found in {model_path}")
                return None
            
            # Find the largest geometry by vertex count
            largest_mesh = max(geometries, key=lambda g: len(g.vertices) if hasattr(g, 'vertices') else 0)
            mesh = largest_mesh
        
        # Center the mesh
        mesh.vertices -= mesh.centroid
        
        # Scale to a standard size
        scale = 1.0 / mesh.scale
        mesh.vertices *= scale
        
        # Align to standard orientation (glasses typically have X as width, Y as height, Z as depth)
        # This is a simplified approach - in practice, you might need more sophisticated alignment
        # based on principal component analysis or specific features of glasses
        
        # Save the normalized model
        mesh.export(output_path)
        
        return output_path
    except Exception as e:
        print(f"Error normalizing model {model_path}: {e}")
        return None

def render_model_views(model_path, output_dir, num_views=4):
    """Render multiple views of a 3D model"""
    try:
        # Load the model
        mesh = trimesh.load(model_path)
        
        # Create a scene
        scene = trimesh.Scene(mesh)
        
        # Get the base filename
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # Render views from different angles
        renders = []
        for i in range(num_views):
            # Rotation angle in degrees
            angle = i * (360 / num_views)
            
            # Create a rotation matrix around the Y axis
            rotation = trimesh.transformations.rotation_matrix(
                angle * np.pi / 180, [0, 1, 0])
            
            # Apply rotation
            rotated_scene = scene.copy()
            rotated_scene.apply_transform(rotation)
            
            # Render the scene
            render_path = os.path.join(output_dir, f"{base_name}_view_{i}.png")
            rotated_scene.save_image(render_path, resolution=(512, 512))
            
            renders.append(render_path)
        
        return renders
    except Exception as e:
        print(f"Error rendering model {model_path}: {e}")
        return []

def pair_models_with_images(models_dir, images_dir, output_dir):
    """Pair 3D models with corresponding 2D images based on similarity"""
    # This is a placeholder - in a real implementation, you would use
    # image similarity or feature matching to pair models with images
    
    # Get list of models
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.glb')]
    
    # Get list of images
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    
    # Create pairs (randomly for this placeholder)
    pairs = []
    
    # Use the minimum of the two lists to ensure we have pairs
    num_pairs = min(len(model_files), len(image_files))
    
    for i in range(num_pairs):
        model_file = model_files[i]
        image_file = image_files[i]
        
        model_path = os.path.join(models_dir, model_file)
        image_path = os.path.join(images_dir, image_file)
        
        # Create a unique pair ID
        pair_id = hashlib.md5((model_file + image_file).encode()).hexdigest()
        
        # Copy files to paired directory
        paired_model_path = os.path.join(output_dir, f"{pair_id}_model.glb")
        paired_image_path = os.path.join(output_dir, f"{pair_id}_image.png")
        
        shutil.copy(model_path, paired_model_path)
        
        # Convert image to PNG if needed
        img = Image.open(image_path)
        img.save(paired_image_path)
        
        pairs.append({
            "id": pair_id,
            "model_path": paired_model_path,
            "image_path": paired_image_path,
            "original_model": model_file,
            "original_image": image_file
        })
    
    return pairs

def main(args):
    """Main function for collecting ground truth 3D models"""
    # Create directory structure
    create_directory_structure(args.output_dir)
    
    # Get API keys from environment variables
    sketchfab_api_key = os.environ.get(SOURCES["sketchfab"]["api_key_env"])
    
    # Search for models
    models = []
    
    if args.source == "sketchfab" or args.source == "all":
        sketchfab_models = search_sketchfab_models(
            args.query,
            sketchfab_api_key,
            max_models=args.max_models
        )
        models.extend(sketchfab_models)
    
    # Download models
    raw_dir = os.path.join(args.output_dir, "raw")
    downloaded_models = []
    
    print(f"Downloading {len(models)} models...")
    for model in tqdm(models, desc="Downloading models"):
        if model["source"] == "sketchfab":
            model_path = download_sketchfab_model(
                model["id"],
                sketchfab_api_key,
                raw_dir
            )
            if model_path:
                model["file_path"] = model_path
                downloaded_models.append(model)
    
    print(f"Successfully downloaded {len(downloaded_models)} models")
    
    # Normalize models
    processed_dir = os.path.join(args.output_dir, "processed")
    normalized_models = []
    
    print("Normalizing models...")
    for model in tqdm(downloaded_models, desc="Normalizing models"):
        if "file_path" in model:
            output_path = os.path.join(processed_dir, f"normalized_{os.path.basename(model['file_path'])}")
            normalized_path = normalize_model(model["file_path"], output_path)
            if normalized_path:
                model["normalized_path"] = normalized_path
                normalized_models.append(model)
    
    print(f"Successfully normalized {len(normalized_models)} models")
    
    # Render model views
    renders_dir = os.path.join(args.output_dir, "renders")
    
    print("Rendering model views...")
    for model in tqdm(normalized_models, desc="Rendering models"):
        if "normalized_path" in model:
            render_paths = render_model_views(model["normalized_path"], renders_dir)
            if render_paths:
                model["render_paths"] = render_paths
    
    # Pair models with images if image directory is provided
    if args.images_dir:
        paired_dir = os.path.join(args.output_dir, "paired")
        
        print("Pairing models with images...")
        pairs = pair_models_with_images(
            processed_dir,
            args.images_dir,
            paired_dir
        )
        
        print(f"Created {len(pairs)} model-image pairs")
        
        # Save pairs metadata
        pairs_path = os.path.join(args.output_dir, "pairs.json")
        with open(pairs_path, "w") as f:
            json.dump(pairs, f, indent=2)
    
    # Save models metadata
    metadata_path = os.path.join(args.output_dir, "models_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(normalized_models, f, indent=2)
    
    print(f"Ground truth collection complete. Metadata saved to {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect ground truth 3D models for evaluation")
    parser.add_argument("--output_dir", type=str, default="ground_truth",
                        help="Output directory for ground truth dataset")
    parser.add_argument("--query", type=str, default="eyeglasses glasses spectacles",
                        help="Search query for 3D models")
    parser.add_argument("--max_models", type=int, default=50,
                        help="Maximum number of models to download")
    parser.add_argument("--source", type=str, default="all",
                        choices=["all", "sketchfab", "turbosquid", "cgtrader"],
                        help="Source to download from")
    parser.add_argument("--images_dir", type=str,
                        help="Directory containing 2D images to pair with models")
    
    args = parser.parse_args()
    main(args)
