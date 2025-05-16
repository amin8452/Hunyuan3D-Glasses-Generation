#!/usr/bin/env python
"""
Script for collecting and organizing a diverse dataset of glasses images.
This script can:
1. Download images from various sources
2. Organize and categorize them
3. Apply basic filtering to ensure quality
4. Create a balanced dataset across different styles
"""

import os
import argparse
import requests
import shutil
import json
import random
import time
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Sources for glasses images
SOURCES = {
    "eyebuydirect": "https://www.eyebuydirect.com/eyeglasses",
    "warbyparker": "https://www.warbyparker.com/eyeglasses",
    "zenni": "https://www.zennioptical.com/glasses",
    "glassesusa": "https://www.glassesusa.com/eyeglasses",
    "rayban": "https://www.ray-ban.com/usa/eyeglasses",
    "persol": "https://www.persol.com/usa/eyeglasses"
}

# Categories of glasses
CATEGORIES = [
    "round", "square", "rectangle", "oval", "cat-eye", "aviator", "wayfarer", 
    "browline", "rimless", "semi-rimless", "oversized", "geometric"
]

# Materials
MATERIALS = [
    "metal", "plastic", "acetate", "titanium", "wood", "carbon-fiber"
]

# Colors
COLORS = [
    "black", "brown", "gold", "silver", "blue", "red", "green", "tortoise", 
    "clear", "multicolor"
]

def create_directory_structure(base_dir):
    """Create the directory structure for the dataset"""
    # Main directories
    os.makedirs(os.path.join(base_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "processed"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test", "images"), exist_ok=True)
    
    # Category directories within raw
    for category in CATEGORIES:
        os.makedirs(os.path.join(base_dir, "raw", category), exist_ok=True)
    
    print(f"Created directory structure in {base_dir}")

def download_image(url, save_path):
    """Download an image from a URL and save it to a path"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        # Check if the content is an image
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image'):
            return False, f"Not an image: {content_type}"
        
        # Open the image to verify it's valid
        img = Image.open(BytesIO(response.content))
        
        # Save the image
        with open(save_path, 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
        
        return True, save_path
    except Exception as e:
        return False, str(e)

def download_images_from_source(source_name, source_url, output_dir, num_images=100):
    """
    Download images from a specific source.
    This is a placeholder - in a real implementation, you would need to use
    web scraping or APIs specific to each source.
    """
    print(f"Downloading images from {source_name}...")
    
    # This is where you would implement source-specific scraping logic
    # For demonstration, we'll just create a placeholder message
    placeholder_path = os.path.join(output_dir, f"{source_name}_placeholder.txt")
    with open(placeholder_path, 'w') as f:
        f.write(f"To download from {source_name} ({source_url}), you would need to:\n")
        f.write("1. Check the website's terms of service and robots.txt\n")
        f.write("2. Use appropriate scraping libraries (BeautifulSoup, Selenium, etc.)\n")
        f.write("3. Implement pagination and image extraction logic\n")
        f.write("4. Respect rate limits and implement proper error handling\n")
    
    print(f"Created placeholder at {placeholder_path}")
    print("Note: Actual implementation would require source-specific scraping logic")
    
    return []

def is_glasses_image(image_path, confidence_threshold=0.7):
    """
    Check if an image contains glasses using basic computer vision.
    This is a simplified placeholder - a real implementation would use
    a trained classifier or object detection model.
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for contours that might be glasses
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by shape and size
        glasses_like_contours = 0
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            if area < 1000:  # Too small
                continue
                
            # Check aspect ratio (glasses are typically wider than tall)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 1.5 < aspect_ratio < 4.0:  # Typical glasses aspect ratio
                glasses_like_contours += 1
        
        # Simple heuristic: if we have some contours that look like glasses, it might be a glasses image
        # This is very simplified and would need to be replaced with a proper classifier
        confidence = min(1.0, glasses_like_contours / 3.0)
        
        return confidence > confidence_threshold
    except Exception:
        return False

def categorize_image(image_path):
    """
    Categorize an image of glasses based on its visual characteristics.
    This is a placeholder - a real implementation would use a trained classifier.
    """
    # In a real implementation, you would:
    # 1. Use a pre-trained classifier to identify the style
    # 2. Extract features to determine material and color
    # 3. Return the most likely category
    
    # For demonstration, we'll randomly assign a category
    category = random.choice(CATEGORIES)
    material = random.choice(MATERIALS)
    color = random.choice(COLORS)
    
    return {
        "category": category,
        "material": material,
        "color": color
    }

def process_image(image_path, output_dir, target_size=(512, 512)):
    """Process an image for the dataset"""
    try:
        # Check if it's a glasses image
        if not is_glasses_image(image_path):
            return None, "Not a glasses image"
        
        # Categorize the image
        metadata = categorize_image(image_path)
        category = metadata["category"]
        
        # Create a unique filename based on content hash
        with open(image_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # Process the image
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size, Image.LANCZOS)
        
        # Save to category directory
        category_dir = os.path.join(output_dir, "raw", category)
        processed_path = os.path.join(category_dir, f"{file_hash}.png")
        img.save(processed_path)
        
        # Add to metadata
        metadata["file_path"] = processed_path
        metadata["original_path"] = image_path
        metadata["hash"] = file_hash
        
        return metadata, None
    except Exception as e:
        return None, str(e)

def split_dataset(metadata_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, base_dir="data"):
    """Split the dataset into train, validation, and test sets"""
    # Shuffle the metadata list
    random.shuffle(metadata_list)
    
    # Calculate split indices
    n = len(metadata_list)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split the data
    train_data = metadata_list[:train_end]
    val_data = metadata_list[train_end:val_end]
    test_data = metadata_list[val_end:]
    
    # Create directories if they don't exist
    train_dir = os.path.join(base_dir, "train", "images")
    val_dir = os.path.join(base_dir, "val", "images")
    test_dir = os.path.join(base_dir, "test", "images")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Copy images to their respective directories
    splits = {
        "train": (train_data, train_dir),
        "val": (val_data, val_dir),
        "test": (test_data, test_dir)
    }
    
    for split_name, (data, directory) in splits.items():
        print(f"Processing {split_name} split ({len(data)} images)...")
        
        # Copy images and update metadata
        for item in tqdm(data):
            src_path = item["file_path"]
            filename = f"{item['hash']}.png"
            dst_path = os.path.join(directory, filename)
            
            # Copy the image
            shutil.copy(src_path, dst_path)
            
            # Update the metadata
            item["split"] = split_name
            item["dataset_path"] = dst_path
    
    # Save the metadata
    metadata_path = os.path.join(base_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "train": [item for item in metadata_list if item.get("split") == "train"],
            "val": [item for item in metadata_list if item.get("split") == "val"],
            "test": [item for item in metadata_list if item.get("split") == "test"]
        }, f, indent=2)
    
    print(f"Dataset split complete. Metadata saved to {metadata_path}")
    print(f"Train: {len(train_data)} images")
    print(f"Validation: {len(val_data)} images")
    print(f"Test: {len(test_data)} images")

def main(args):
    """Main function for data collection"""
    # Create directory structure
    create_directory_structure(args.output_dir)
    
    # Download images from sources
    all_images = []
    for source_name, source_url in SOURCES.items():
        images = download_images_from_source(
            source_name, 
            source_url, 
            os.path.join(args.output_dir, "raw"),
            args.num_images_per_source
        )
        all_images.extend(images)
    
    # If images were provided directly, add them
    if args.input_dir:
        print(f"Processing images from {args.input_dir}...")
        for root, _, files in os.walk(args.input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(root, file))
    
    # Process images
    print(f"Processing {len(all_images)} images...")
    metadata_list = []
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for image_path in all_images:
            future = executor.submit(process_image, image_path, args.output_dir)
            futures.append(future)
        
        for future in tqdm(futures):
            metadata, error = future.result()
            if metadata:
                metadata_list.append(metadata)
            elif error:
                print(f"Error processing image: {error}")
    
    # Split the dataset
    split_dataset(
        metadata_list,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.output_dir
    )
    
    print("Data collection and processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and organize a dataset of glasses images")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for the dataset")
    parser.add_argument("--input_dir", type=str, help="Directory containing existing images to process")
    parser.add_argument("--num_images_per_source", type=int, default=100, help="Number of images to download from each source")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of images for training")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of images for validation")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of images for testing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads")
    
    args = parser.parse_args()
    main(args)
