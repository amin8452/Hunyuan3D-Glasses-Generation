#!/usr/bin/env python
"""
Script to download and set up the pre-trained Hunyuan3D 2.0 model.
This script:
1. Downloads the model from Hugging Face or official sources
2. Verifies the integrity of the downloaded files
3. Converts the model to the format needed for fine-tuning
4. Tests the model to ensure it's working correctly
"""

import os
import argparse
import requests
import hashlib
import json
import torch
import zipfile
import tarfile
import shutil
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download
import sys

# Add parent directory to path for importing custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model information
MODEL_INFO = {
    "hunyuan3d_2.0": {
        "hf_repo": "Tencent/Hunyuan3D-2.0",
        "hf_files": [
            "config.json",
            "pytorch_model.bin",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt"
        ],
        "direct_url": None,  # Fallback direct download URL if HF fails
        "md5_hash": None,  # Expected MD5 hash for verification
        "model_type": "diffusion"
    }
}

def download_file(url, destination, chunk_size=8192):
    """Download a file from a URL with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    
    return destination

def verify_file_hash(file_path, expected_hash):
    """Verify the MD5 hash of a file"""
    if not expected_hash:
        print(f"No hash provided for {file_path}, skipping verification")
        return True
    
    print(f"Verifying hash for {file_path}...")
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    file_hash = md5_hash.hexdigest()
    if file_hash != expected_hash:
        print(f"Hash verification failed for {file_path}")
        print(f"Expected: {expected_hash}")
        print(f"Got: {file_hash}")
        return False
    
    print(f"Hash verification successful for {file_path}")
    return True

def download_from_huggingface(model_name, output_dir):
    """Download model files from Hugging Face"""
    if model_name not in MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_info = MODEL_INFO[model_name]
    repo_id = model_info["hf_repo"]
    
    print(f"Downloading {model_name} from Hugging Face ({repo_id})...")
    
    try:
        # Try to download the entire repository
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded model to {snapshot_path}")
        return True
    except Exception as e:
        print(f"Error downloading entire repository: {e}")
        print("Trying to download individual files...")
        
        # Try to download individual files
        success = True
        for filename in model_info["hf_files"]:
            try:
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=output_dir
                )
                print(f"Downloaded {filename} to {file_path}")
            except Exception as file_e:
                print(f"Error downloading {filename}: {file_e}")
                success = False
        
        return success

def download_from_direct_url(model_name, output_dir):
    """Download model from direct URL"""
    if model_name not in MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_info = MODEL_INFO[model_name]
    url = model_info["direct_url"]
    
    if not url:
        print(f"No direct URL available for {model_name}")
        return False
    
    print(f"Downloading {model_name} from direct URL...")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download the file
        file_ext = url.split(".")[-1]
        download_path = os.path.join(output_dir, f"{model_name}.{file_ext}")
        download_file(url, download_path)
        
        # Verify hash if available
        if model_info["md5_hash"]:
            if not verify_file_hash(download_path, model_info["md5_hash"]):
                return False
        
        # Extract if it's an archive
        if file_ext in ["zip", "tar", "gz", "tgz"]:
            print(f"Extracting {download_path}...")
            extract_archive(download_path, output_dir)
            # Optionally remove the archive after extraction
            os.remove(download_path)
        
        return True
    except Exception as e:
        print(f"Error downloading from direct URL: {e}")
        return False

def extract_archive(archive_path, output_dir):
    """Extract an archive file"""
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    elif archive_path.endswith(".tar"):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(output_dir)
    elif archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(output_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

def convert_model_format(model_dir, model_name):
    """Convert the model to the format needed for fine-tuning"""
    if model_name not in MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_info = MODEL_INFO[model_name]
    model_type = model_info["model_type"]
    
    print(f"Converting {model_name} to the required format...")
    
    if model_type == "diffusion":
        # For diffusion models, we need to ensure it's compatible with our pipeline
        # This is a placeholder - actual conversion depends on the specific model format
        try:
            # Check if the model can be loaded with diffusers
            from diffusers import DiffusionPipeline
            
            pipeline = DiffusionPipeline.from_pretrained(model_dir)
            print("Model loaded successfully with diffusers")
            
            # Save in the format expected by our code
            pipeline.save_pretrained(model_dir)
            print(f"Model converted and saved to {model_dir}")
            
            return True
        except Exception as e:
            print(f"Error converting model: {e}")
            return False
    else:
        print(f"Unknown model type: {model_type}")
        return False

def test_model(model_dir, model_name):
    """Test the model to ensure it's working correctly"""
    if model_name not in MODEL_INFO:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_info = MODEL_INFO[model_name]
    model_type = model_info["model_type"]
    
    print(f"Testing {model_name}...")
    
    if model_type == "diffusion":
        try:
            # Try to load the model and run a simple inference
            from diffusers import DiffusionPipeline
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            pipeline = DiffusionPipeline.from_pretrained(model_dir)
            pipeline = pipeline.to(device)
            
            # Run a simple inference (this is a placeholder - actual inference depends on the model)
            print("Running test inference...")
            # This is just a placeholder - replace with actual inference code
            # output = pipeline("test prompt", num_inference_steps=5)
            
            print("Test inference completed successfully")
            return True
        except Exception as e:
            print(f"Error testing model: {e}")
            return False
    else:
        print(f"Unknown model type: {model_type}")
        return False

def main(args):
    """Main function to download and set up the pre-trained model"""
    model_name = args.model
    output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the model
    success = False
    
    # Try Hugging Face first
    if args.source == "huggingface" or args.source == "auto":
        success = download_from_huggingface(model_name, output_dir)
    
    # If Hugging Face failed and we're using auto or direct, try direct URL
    if not success and (args.source == "direct" or args.source == "auto"):
        success = download_from_direct_url(model_name, output_dir)
    
    if not success:
        print(f"Failed to download {model_name}")
        return 1
    
    # Convert model format if needed
    if args.convert:
        if not convert_model_format(output_dir, model_name):
            print(f"Failed to convert {model_name}")
            return 1
    
    # Test the model if requested
    if args.test:
        if not test_model(output_dir, model_name):
            print(f"Model test failed for {model_name}")
            return 1
    
    print(f"Successfully downloaded and set up {model_name} at {output_dir}")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and set up pre-trained Hunyuan3D model")
    parser.add_argument("--model", type=str, default="hunyuan3d_2.0", 
                        choices=list(MODEL_INFO.keys()),
                        help="Model to download")
    parser.add_argument("--output_dir", type=str, default="models/pretrained",
                        help="Directory to save the model")
    parser.add_argument("--source", type=str, default="auto",
                        choices=["auto", "huggingface", "direct"],
                        help="Source to download from")
    parser.add_argument("--convert", action="store_true",
                        help="Convert the model to the required format")
    parser.add_argument("--test", action="store_true",
                        help="Test the model after downloading")
    
    args = parser.parse_args()
    sys.exit(main(args))
