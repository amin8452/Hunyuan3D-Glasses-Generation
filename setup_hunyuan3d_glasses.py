#!/usr/bin/env python
"""
Master script to set up the complete Hunyuan3D glasses generation pipeline.
This script:
1. Collects data using APIs
2. Downloads and sets up the pre-trained model
3. Configures GPU resources
4. Collects ground truth 3D models
5. Prepares the environment for training and evaluation
"""

import os
import argparse
import subprocess
import json
import time
from datetime import datetime
import sys

def run_command(cmd, description):
    """Run a command and print its output"""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}\n")
    print(f"Running command: {' '.join(cmd)}")
    
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    end_time = time.time()
    
    print(f"\nCommand completed in {end_time - start_time:.2f} seconds")
    print(f"Return code: {process.returncode}")
    
    if process.returncode != 0:
        print(f"WARNING: Command failed with return code {process.returncode}")
    
    return process.returncode

def setup_environment():
    """Set up the Python environment with required packages"""
    print("Setting up Python environment...")
    
    # Install required packages
    cmd = [
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ]
    
    return run_command(cmd, "Installing dependencies")

def collect_data(args):
    """Collect data using APIs"""
    print("Collecting data using APIs...")
    
    cmd = [
        sys.executable, "scripts/api_data_collection.py",
        "--output_dir", args.data_dir,
        "--query", args.query,
        "--max_images", str(args.max_images),
        "--max_models", str(args.max_models)
    ]
    
    return run_command(cmd, "Data collection")

def download_pretrained_model(args):
    """Download and set up the pre-trained model"""
    print("Downloading pre-trained model...")
    
    cmd = [
        sys.executable, "scripts/download_pretrained_model.py",
        "--model", args.model,
        "--output_dir", args.model_dir,
        "--source", args.model_source
    ]
    
    if args.convert_model:
        cmd.append("--convert")
    
    if args.test_model:
        cmd.append("--test")
    
    return run_command(cmd, "Downloading pre-trained model")

def setup_gpu_resources(args):
    """Check and configure GPU resources"""
    print("Setting up GPU resources...")
    
    cmd = [
        sys.executable, "scripts/gpu_setup.py",
        "--output", os.path.join(args.output_dir, "gpu_config.json"),
        "--dataset_size", str(args.dataset_size),
        "--epochs", str(args.epochs)
    ]
    
    return run_command(cmd, "GPU setup")

def collect_ground_truth(args):
    """Collect ground truth 3D models"""
    print("Collecting ground truth 3D models...")
    
    cmd = [
        sys.executable, "scripts/collect_ground_truth.py",
        "--output_dir", args.ground_truth_dir,
        "--query", args.query,
        "--max_models", str(args.max_models),
        "--source", args.model_source
    ]
    
    if args.data_dir:
        cmd.extend(["--images_dir", os.path.join(args.data_dir, "train", "images")])
    
    return run_command(cmd, "Ground truth collection")

def prepare_training(args):
    """Prepare for training by creating necessary directories and files"""
    print("Preparing for training...")
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "evaluation"), exist_ok=True)
    
    # Load GPU configuration if available
    gpu_config_path = os.path.join(args.output_dir, "gpu_config.json")
    training_args = {}
    
    if os.path.exists(gpu_config_path):
        try:
            with open(gpu_config_path, "r") as f:
                gpu_config = json.load(f)
            
            training_args = gpu_config.get("recommended_training_args", {})
            print("Loaded recommended training arguments from GPU configuration")
        except Exception as e:
            print(f"Error loading GPU configuration: {e}")
    
    # Create a training configuration file
    training_config = {
        "data": {
            "train_data": os.path.join(args.data_dir, "train"),
            "val_data": os.path.join(args.data_dir, "val"),
            "test_data": os.path.join(args.data_dir, "test")
        },
        "model": {
            "pretrained_model": os.path.join(args.model_dir, args.model),
            "output_dir": os.path.join(args.output_dir, "models")
        },
        "training": {
            "batch_size": training_args.get("batch_size", 4),
            "epochs": args.epochs,
            "learning_rate": training_args.get("learning_rate", 1e-5),
            "gradient_accumulation_steps": training_args.get("gradient_accumulation_steps", 1),
            "use_amp": training_args.get("use_amp", False),
            "num_workers": training_args.get("num_workers", 4)
        },
        "evaluation": {
            "ground_truth_dir": args.ground_truth_dir,
            "output_dir": os.path.join(args.output_dir, "evaluation")
        }
    }
    
    # Save the training configuration
    training_config_path = os.path.join(args.output_dir, "training_config.json")
    with open(training_config_path, "w") as f:
        json.dump(training_config, f, indent=2)
    
    print(f"Training configuration saved to {training_config_path}")
    
    # Create a README file with instructions
    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"""# Hunyuan3D Glasses Generation Setup

Setup completed on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Data

- Training data: {os.path.join(args.data_dir, "train")}
- Validation data: {os.path.join(args.data_dir, "val")}
- Test data: {os.path.join(args.data_dir, "test")}

## Pre-trained Model

- Model: {args.model}
- Location: {os.path.join(args.model_dir, args.model)}

## Ground Truth

- Ground truth models: {args.ground_truth_dir}

## Next Steps

1. **Fine-tune the model**:
   ```bash
   python scripts/fine_tuning.py --config {training_config_path}
   ```

2. **Optimize hyperparameters**:
   ```bash
   python scripts/hyperparameter_optimization.py --config {training_config_path}
   ```

3. **Evaluate the model**:
   ```bash
   python scripts/comprehensive_evaluation.py --config {training_config_path}
   ```

4. **Generate glasses from images**:
   ```bash
   python scripts/generate.py --model_path models/fine_tuned/best_model.pt --input_image your_image.jpg --output_model glasses.glb
   ```

## GPU Configuration

Recommended settings based on your hardware:
- Batch size: {training_args.get("batch_size", 4)}
- Learning rate: {training_args.get("learning_rate", 1e-5)}
- Gradient accumulation steps: {training_args.get("gradient_accumulation_steps", 1)}
- Mixed precision: {training_args.get("use_amp", False)}
- Number of workers: {training_args.get("num_workers", 4)}
""")
    
    print(f"README with instructions saved to {readme_path}")
    return 0

def main(args):
    """Main function to set up the Hunyuan3D glasses generation pipeline"""
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up environment
    if args.setup_env:
        setup_environment()
    
    # Collect data
    if args.collect_data:
        collect_data(args)
    
    # Download pre-trained model
    if args.download_model:
        download_pretrained_model(args)
    
    # Set up GPU resources
    if args.setup_gpu:
        setup_gpu_resources(args)
    
    # Collect ground truth
    if args.collect_ground_truth:
        collect_ground_truth(args)
    
    # Prepare for training
    if args.prepare_training:
        prepare_training(args)
    
    print("\nSetup completed successfully!")
    print(f"All outputs are available in: {args.output_dir}")
    print(f"See {os.path.join(args.output_dir, 'README.md')} for next steps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up the Hunyuan3D glasses generation pipeline")
    
    # Setup steps
    parser.add_argument("--setup_env", action="store_true", help="Set up Python environment")
    parser.add_argument("--collect_data", action="store_true", help="Collect data using APIs")
    parser.add_argument("--download_model", action="store_true", help="Download pre-trained model")
    parser.add_argument("--setup_gpu", action="store_true", help="Set up GPU resources")
    parser.add_argument("--collect_ground_truth", action="store_true", help="Collect ground truth 3D models")
    parser.add_argument("--prepare_training", action="store_true", help="Prepare for training")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Directory for dataset")
    parser.add_argument("--query", type=str, default="eyeglasses glasses spectacles", help="Search query")
    parser.add_argument("--max_images", type=int, default=200, help="Maximum number of images to download")
    parser.add_argument("--max_models", type=int, default=50, help="Maximum number of 3D models to download")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="hunyuan3d_2.0", help="Model to download")
    parser.add_argument("--model_dir", type=str, default="models/pretrained", help="Directory for pre-trained model")
    parser.add_argument("--model_source", type=str, default="auto", 
                        choices=["auto", "huggingface", "direct", "all", "sketchfab"],
                        help="Source to download from")
    parser.add_argument("--convert_model", action="store_true", help="Convert the model to the required format")
    parser.add_argument("--test_model", action="store_true", help="Test the model after downloading")
    
    # Ground truth arguments
    parser.add_argument("--ground_truth_dir", type=str, default="ground_truth", 
                        help="Directory for ground truth 3D models")
    
    # Training arguments
    parser.add_argument("--dataset_size", type=int, default=1000, help="Number of samples in the dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="hunyuan3d_glasses_setup",
                        help="Output directory for setup results")
    
    args = parser.parse_args()
    
    # If all is specified, run all steps
    if args.all:
        args.setup_env = True
        args.collect_data = True
        args.download_model = True
        args.setup_gpu = True
        args.collect_ground_truth = True
        args.prepare_training = True
    
    # Ensure at least one step is selected
    if not any([args.setup_env, args.collect_data, args.download_model, 
                args.setup_gpu, args.collect_ground_truth, args.prepare_training]):
        parser.error("At least one setup step must be selected")
    
    main(args)
