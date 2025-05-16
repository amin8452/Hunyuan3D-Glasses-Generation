#!/usr/bin/env python
"""
Script to run the entire Hunyuan3D glasses generation pipeline:
1. Data collection
2. Hyperparameter optimization
3. Fine-tuning
4. Evaluation
"""

import os
import argparse
import subprocess
import time
from datetime import datetime

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

def main(args):
    """Run the entire pipeline"""
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    
    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    data_dir = os.path.join(run_dir, "data")
    models_dir = os.path.join(run_dir, "models")
    hyperopt_dir = os.path.join(run_dir, "hyperopt")
    eval_dir = os.path.join(run_dir, "evaluation")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(hyperopt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Step 1: Data Collection
    if args.run_data_collection:
        data_cmd = [
            "python", "scripts/data_collection.py",
            "--output_dir", data_dir
        ]
        
        if args.input_dir:
            data_cmd.extend(["--input_dir", args.input_dir])
        
        run_command(data_cmd, "Data Collection")
    
    # Step 2: Hyperparameter Optimization
    if args.run_hyperopt:
        hyperopt_cmd = [
            "python", "scripts/hyperparameter_optimization.py",
            "--train_data", os.path.join(data_dir, "train"),
            "--val_data", os.path.join(data_dir, "val"),
            "--pretrained_model", args.pretrained_model,
            "--n_trials", str(args.n_trials),
            "--epochs", str(args.hyperopt_epochs),
            "--output_dir", hyperopt_dir
        ]
        
        run_command(hyperopt_cmd, "Hyperparameter Optimization")
        
        # Find the best hyperparameters
        import json
        import glob
        
        hyperopt_results = glob.glob(os.path.join(hyperopt_dir, "*_results.json"))
        if hyperopt_results:
            with open(hyperopt_results[0], "r") as f:
                results = json.load(f)
            
            best_params = results.get("best_params", {})
            print(f"Best hyperparameters: {best_params}")
        else:
            print("Warning: No hyperparameter optimization results found")
            best_params = {}
    else:
        best_params = {}
    
    # Step 3: Fine-tuning
    if args.run_fine_tuning:
        fine_tuning_cmd = [
            "python", "scripts/fine_tuning.py",
            "--train_data", os.path.join(data_dir, "train"),
            "--val_data", os.path.join(data_dir, "val"),
            "--pretrained_model", args.pretrained_model,
            "--epochs", str(args.fine_tuning_epochs),
            "--output_dir", models_dir
        ]
        
        # Add best hyperparameters if available
        if "batch_size" in best_params:
            fine_tuning_cmd.extend(["--batch_size", str(best_params["batch_size"])])
        if "lr" in best_params:
            fine_tuning_cmd.extend(["--lr", str(best_params["lr"])])
        if "weight_decay" in best_params:
            fine_tuning_cmd.extend(["--weight_decay", str(best_params["weight_decay"])])
        if "optimizer" in best_params:
            fine_tuning_cmd.extend(["--optimizer", best_params["optimizer"].lower()])
        if "scheduler" in best_params:
            fine_tuning_cmd.extend(["--scheduler", best_params["scheduler"]])
        if "fine_tuning_strategy" in best_params:
            fine_tuning_cmd.extend(["--fine_tuning_strategy", best_params["fine_tuning_strategy"]])
        
        run_command(fine_tuning_cmd, "Fine-tuning")
    
    # Step 4: Evaluation
    if args.run_evaluation:
        # Find the best model
        best_model_path = os.path.join(models_dir, "best_model.pt")
        
        if not os.path.exists(best_model_path):
            print(f"Warning: Best model not found at {best_model_path}")
            print("Searching for latest checkpoint...")
            
            import glob
            checkpoints = sorted(glob.glob(os.path.join(models_dir, "checkpoint_epoch_*.pt")))
            if checkpoints:
                best_model_path = checkpoints[-1]
                print(f"Using latest checkpoint: {best_model_path}")
            else:
                print("No checkpoints found. Skipping evaluation.")
                return
        
        eval_cmd = [
            "python", "scripts/comprehensive_evaluation.py",
            "--model_path", best_model_path,
            "--test_data", os.path.join(data_dir, "test"),
            "--output_dir", eval_dir
        ]
        
        run_command(eval_cmd, "Comprehensive Evaluation")
    
    print("\nPipeline completed successfully!")
    print(f"Results are available in: {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Hunyuan3D glasses generation pipeline")
    
    # Pipeline steps
    parser.add_argument("--run_data_collection", action="store_true", help="Run data collection step")
    parser.add_argument("--run_hyperopt", action="store_true", help="Run hyperparameter optimization step")
    parser.add_argument("--run_fine_tuning", action="store_true", help="Run fine-tuning step")
    parser.add_argument("--run_evaluation", action="store_true", help="Run evaluation step")
    parser.add_argument("--run_all", action="store_true", help="Run all steps")
    
    # Data arguments
    parser.add_argument("--input_dir", type=str, help="Directory containing existing images to process")
    
    # Model arguments
    parser.add_argument("--pretrained_model", type=str, default="models/hunyuan3d_2.0.pt",
                        help="Path to pre-trained Hunyuan3D model")
    
    # Hyperparameter optimization arguments
    parser.add_argument("--n_trials", type=int, default=10, help="Number of hyperparameter optimization trials")
    parser.add_argument("--hyperopt_epochs", type=int, default=5, 
                        help="Number of epochs per hyperparameter optimization trial")
    
    # Fine-tuning arguments
    parser.add_argument("--fine_tuning_epochs", type=int, default=50, help="Number of fine-tuning epochs")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="pipeline_runs",
                        help="Output directory for pipeline results")
    
    args = parser.parse_args()
    
    # If run_all is specified, run all steps
    if args.run_all:
        args.run_data_collection = True
        args.run_hyperopt = True
        args.run_fine_tuning = True
        args.run_evaluation = True
    
    # Ensure at least one step is selected
    if not any([args.run_data_collection, args.run_hyperopt, args.run_fine_tuning, args.run_evaluation]):
        parser.error("At least one pipeline step must be selected")
    
    main(args)
