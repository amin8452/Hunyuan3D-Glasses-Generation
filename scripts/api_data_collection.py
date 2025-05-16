#!/usr/bin/env python
"""
Script for collecting glasses images and 3D models using various APIs.
This script uses real APIs to collect data from:
1. Pexels API for high-quality glasses images
2. Unsplash API for diverse glasses photos
3. Sketchfab API for 3D glasses models
4. Google Poly API for additional 3D models (if available)
"""

import os
import argparse
import requests
import json
import time
import shutil
from PIL import Image
from io import BytesIO
import zipfile
import gzip
import tarfile
import random
from tqdm import tqdm
import concurrent.futures
import hashlib

# API Keys (replace with your own)
PEXELS_API_KEY = "YOUR_PEXELS_API_KEY"  # Get from https://www.pexels.com/api/
UNSPLASH_API_KEY = "YOUR_UNSPLASH_API_KEY"  # Get from https://unsplash.com/developers
SKETCHFAB_API_KEY = "YOUR_SKETCHFAB_API_KEY"  # Get from https://sketchfab.com/settings/password
POLY_API_KEY = "YOUR_POLY_API_KEY"  # Note: Google Poly was shut down, included for reference

# API Endpoints
PEXELS_API_URL = "https://api.pexels.com/v1/search"
UNSPLASH_API_URL = "https://api.unsplash.com/search/photos"
SKETCHFAB_API_URL = "https://api.sketchfab.com/v3/search"
SKETCHFAB_MODEL_URL = "https://api.sketchfab.com/v3/models/{model_id}/download"

def create_directory_structure(base_dir):
    """Create the directory structure for the dataset"""
    # Main directories
    os.makedirs(os.path.join(base_dir, "raw", "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "raw", "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "train", "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "val", "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test", "models"), exist_ok=True)
    
    print(f"Created directory structure in {base_dir}")

def fetch_pexels_images(query, per_page=20, max_images=100):
    """Fetch images from Pexels API"""
    if not PEXELS_API_KEY or PEXELS_API_KEY == "YOUR_PEXELS_API_KEY":
        print("Warning: Pexels API key not set. Skipping Pexels.")
        return []
    
    headers = {"Authorization": PEXELS_API_KEY}
    images = []
    page = 1
    
    with tqdm(total=max_images, desc="Fetching from Pexels") as pbar:
        while len(images) < max_images:
            params = {
                "query": query,
                "per_page": per_page,
                "page": page
            }
            
            try:
                response = requests.get(PEXELS_API_URL, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data.get("photos"):
                    break
                
                for photo in data["photos"]:
                    if len(images) >= max_images:
                        break
                    
                    images.append({
                        "url": photo["src"]["large"],
                        "id": photo["id"],
                        "width": photo["width"],
                        "height": photo["height"],
                        "photographer": photo["photographer"],
                        "source": "pexels",
                        "page_url": photo["url"],
                        "category": "unknown"  # Will be determined later
                    })
                    pbar.update(1)
                
                # Check if there are more pages
                if not data.get("next_page"):
                    break
                
                page += 1
                time.sleep(0.5)  # Respect rate limits
                
            except Exception as e:
                print(f"Error fetching from Pexels: {e}")
                break
    
    print(f"Fetched {len(images)} images from Pexels")
    return images

def fetch_unsplash_images(query, per_page=20, max_images=100):
    """Fetch images from Unsplash API"""
    if not UNSPLASH_API_KEY or UNSPLASH_API_KEY == "YOUR_UNSPLASH_API_KEY":
        print("Warning: Unsplash API key not set. Skipping Unsplash.")
        return []
    
    headers = {"Authorization": f"Client-ID {UNSPLASH_API_KEY}"}
    images = []
    page = 1
    
    with tqdm(total=max_images, desc="Fetching from Unsplash") as pbar:
        while len(images) < max_images:
            params = {
                "query": query,
                "per_page": per_page,
                "page": page
            }
            
            try:
                response = requests.get(UNSPLASH_API_URL, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data.get("results"):
                    break
                
                for photo in data["results"]:
                    if len(images) >= max_images:
                        break
                    
                    images.append({
                        "url": photo["urls"]["regular"],
                        "id": photo["id"],
                        "width": photo["width"],
                        "height": photo["height"],
                        "photographer": photo["user"]["name"],
                        "source": "unsplash",
                        "page_url": photo["links"]["html"],
                        "category": "unknown"  # Will be determined later
                    })
                    pbar.update(1)
                
                # Check if there are more pages
                if page >= data["total_pages"]:
                    break
                
                page += 1
                time.sleep(0.5)  # Respect rate limits
                
            except Exception as e:
                print(f"Error fetching from Unsplash: {e}")
                break
    
    print(f"Fetched {len(images)} images from Unsplash")
    return images

def fetch_sketchfab_models(query, max_models=50):
    """Fetch 3D models from Sketchfab API"""
    if not SKETCHFAB_API_KEY or SKETCHFAB_API_KEY == "YOUR_SKETCHFAB_API_KEY":
        print("Warning: Sketchfab API key not set. Skipping Sketchfab.")
        return []
    
    headers = {"Authorization": f"Token {SKETCHFAB_API_KEY}"}
    models = []
    cursor = None
    
    with tqdm(total=max_models, desc="Fetching from Sketchfab") as pbar:
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
                response = requests.get(SKETCHFAB_API_URL, headers=headers, params=params)
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
                print(f"Error fetching from Sketchfab: {e}")
                break
    
    print(f"Fetched {len(models)} models from Sketchfab")
    return models

def download_sketchfab_model(model_info, output_dir):
    """Download a 3D model from Sketchfab"""
    if not SKETCHFAB_API_KEY or SKETCHFAB_API_KEY == "YOUR_SKETCHFAB_API_KEY":
        return None
    
    headers = {"Authorization": f"Token {SKETCHFAB_API_KEY}"}
    model_id = model_info["id"]
    
    try:
        # Request download URL
        download_url_endpoint = SKETCHFAB_MODEL_URL.format(model_id=model_id)
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
        filename = f"{model_id}_{hashlib.md5(model_info['name'].encode()).hexdigest()[:8]}.zip"
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
            final_path = os.path.join(output_dir, f"{model_id}.glb")
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

def download_image(image_info, output_dir):
    """Download an image from a URL"""
    try:
        response = requests.get(image_info["url"], stream=True, timeout=10)
        response.raise_for_status()
        
        # Check if the content is an image
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image'):
            return None, f"Not an image: {content_type}"
        
        # Create a unique filename
        source = image_info["source"]
        image_id = image_info["id"]
        filename = f"{source}_{image_id}.jpg"
        save_path = os.path.join(output_dir, filename)
        
        # Save the image
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
        
        return save_path, None
    except Exception as e:
        return None, str(e)

def download_images_and_models(image_list, model_list, output_dir, num_workers=4):
    """Download images and models using multiple threads"""
    # Create output directories
    images_dir = os.path.join(output_dir, "raw", "images")
    models_dir = os.path.join(output_dir, "raw", "models")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Download images
    print(f"Downloading {len(image_list)} images...")
    downloaded_images = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_image = {
            executor.submit(download_image, image_info, images_dir): image_info
            for image_info in image_list
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_image), total=len(image_list), desc="Downloading images"):
            image_info = future_to_image[future]
            try:
                path, error = future.result()
                if path:
                    image_info["file_path"] = path
                    downloaded_images.append(image_info)
                elif error:
                    print(f"Error downloading image {image_info['id']}: {error}")
            except Exception as e:
                print(f"Exception downloading image {image_info['id']}: {e}")
    
    print(f"Successfully downloaded {len(downloaded_images)} images")
    
    # Download models
    print(f"Downloading {len(model_list)} 3D models...")
    downloaded_models = []
    
    for model_info in tqdm(model_list, desc="Downloading models"):
        model_path = download_sketchfab_model(model_info, models_dir)
        if model_path:
            model_info["file_path"] = model_path
            downloaded_models.append(model_info)
    
    print(f"Successfully downloaded {len(downloaded_models)} models")
    
    return downloaded_images, downloaded_models

def split_dataset(images, models, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, output_dir="data"):
    """Split the dataset into train, validation, and test sets"""
    # Shuffle the data
    random.shuffle(images)
    random.shuffle(models)
    
    # Calculate split indices for images
    n_images = len(images)
    train_end_img = int(n_images * train_ratio)
    val_end_img = train_end_img + int(n_images * val_ratio)
    
    # Split images
    train_images = images[:train_end_img]
    val_images = images[train_end_img:val_end_img]
    test_images = images[val_end_img:]
    
    # Calculate split indices for models
    n_models = len(models)
    train_end_model = int(n_models * train_ratio)
    val_end_model = train_end_model + int(n_models * val_ratio)
    
    # Split models
    train_models = models[:train_end_model]
    val_models = models[train_end_model:val_end_model]
    test_models = models[val_end_model:]
    
    # Copy images to their respective directories
    for split_name, split_images in [("train", train_images), ("val", val_images), ("test", test_images)]:
        target_dir = os.path.join(output_dir, split_name, "images")
        os.makedirs(target_dir, exist_ok=True)
        
        for img_info in tqdm(split_images, desc=f"Copying {split_name} images"):
            if "file_path" in img_info:
                src_path = img_info["file_path"]
                filename = os.path.basename(src_path)
                dst_path = os.path.join(target_dir, filename)
                
                try:
                    shutil.copy(src_path, dst_path)
                    img_info["dataset_path"] = dst_path
                    img_info["split"] = split_name
                except Exception as e:
                    print(f"Error copying image {filename}: {e}")
    
    # Copy models to their respective directories
    for split_name, split_models in [("train", train_models), ("val", val_models), ("test", test_models)]:
        target_dir = os.path.join(output_dir, split_name, "models")
        os.makedirs(target_dir, exist_ok=True)
        
        for model_info in tqdm(split_models, desc=f"Copying {split_name} models"):
            if "file_path" in model_info:
                src_path = model_info["file_path"]
                filename = os.path.basename(src_path)
                dst_path = os.path.join(target_dir, filename)
                
                try:
                    shutil.copy(src_path, dst_path)
                    model_info["dataset_path"] = dst_path
                    model_info["split"] = split_name
                except Exception as e:
                    print(f"Error copying model {filename}: {e}")
    
    # Create metadata
    metadata = {
        "images": {
            "train": [img for img in images if img.get("split") == "train"],
            "val": [img for img in images if img.get("split") == "val"],
            "test": [img for img in images if img.get("split") == "test"]
        },
        "models": {
            "train": [model for model in models if model.get("split") == "train"],
            "val": [model for model in models if model.get("split") == "val"],
            "test": [model for model in models if model.get("split") == "test"]
        }
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset split complete. Metadata saved to {metadata_path}")
    print(f"Train: {len(train_images)} images, {len(train_models)} models")
    print(f"Validation: {len(val_images)} images, {len(val_models)} models")
    print(f"Test: {len(test_images)} images, {len(test_models)} models")

def main(args):
    """Main function for API data collection"""
    # Create directory structure
    create_directory_structure(args.output_dir)
    
    # Fetch images from APIs
    all_images = []
    
    # Pexels API
    pexels_images = fetch_pexels_images(
        args.query, 
        per_page=args.per_page, 
        max_images=args.max_images // 2
    )
    all_images.extend(pexels_images)
    
    # Unsplash API
    unsplash_images = fetch_unsplash_images(
        args.query, 
        per_page=args.per_page, 
        max_images=args.max_images // 2
    )
    all_images.extend(unsplash_images)
    
    # Fetch 3D models from Sketchfab
    models = fetch_sketchfab_models(args.query, max_models=args.max_models)
    
    # Download images and models
    downloaded_images, downloaded_models = download_images_and_models(
        all_images, 
        models, 
        args.output_dir,
        num_workers=args.num_workers
    )
    
    # Split the dataset
    split_dataset(
        downloaded_images,
        downloaded_models,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        output_dir=args.output_dir
    )
    
    print("API data collection complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect glasses images and 3D models using APIs")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for the dataset")
    parser.add_argument("--query", type=str, default="eyeglasses glasses spectacles", help="Search query for APIs")
    parser.add_argument("--max_images", type=int, default=200, help="Maximum number of images to download")
    parser.add_argument("--max_models", type=int, default=50, help="Maximum number of 3D models to download")
    parser.add_argument("--per_page", type=int, default=20, help="Number of results per page")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of images for training")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of images for validation")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of images for testing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads")
    
    args = parser.parse_args()
    main(args)
