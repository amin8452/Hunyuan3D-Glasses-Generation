#!/usr/bin/env python
"""
Script to check and configure GPU resources for Hunyuan3D training.
This script:
1. Checks available GPU resources
2. Benchmarks GPU performance for 3D model generation
3. Configures optimal settings based on available hardware
4. Estimates training time and resource requirements
"""

import os
import argparse
import json
import time
import torch
import numpy as np
import psutil
import GPUtil
from tqdm import tqdm
import sys

# Add parent directory to path for importing custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_cuda_availability():
    """Check if CUDA is available and return device information"""
    if not torch.cuda.is_available():
        print("CUDA is not available. Training will be very slow on CPU.")
        return {
            "cuda_available": False,
            "device": "cpu",
            "device_count": 0,
            "devices": []
        }
    
    # Get device information
    device_count = torch.cuda.device_count()
    devices = []
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_capability = torch.cuda.get_device_capability(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
        
        devices.append({
            "index": i,
            "name": device_name,
            "compute_capability": f"{device_capability[0]}.{device_capability[1]}",
            "total_memory_gb": total_memory
        })
    
    return {
        "cuda_available": True,
        "device": "cuda",
        "device_count": device_count,
        "devices": devices
    }

def check_system_resources():
    """Check system resources (CPU, RAM)"""
    # CPU information
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    
    # Memory information
    memory = psutil.virtual_memory()
    total_memory_gb = memory.total / (1024 ** 3)
    available_memory_gb = memory.available / (1024 ** 3)
    
    # Disk information
    disk = psutil.disk_usage('/')
    total_disk_gb = disk.total / (1024 ** 3)
    free_disk_gb = disk.free / (1024 ** 3)
    
    return {
        "cpu": {
            "physical_cores": cpu_count,
            "logical_cores": cpu_count_logical
        },
        "memory": {
            "total_gb": total_memory_gb,
            "available_gb": available_memory_gb,
            "percent_used": memory.percent
        },
        "disk": {
            "total_gb": total_disk_gb,
            "free_gb": free_disk_gb,
            "percent_used": disk.percent
        }
    }

def benchmark_gpu_memory(device, sizes=[512, 1024, 2048, 4096]):
    """Benchmark GPU memory usage for different tensor sizes"""
    if device == "cpu":
        print("Skipping GPU memory benchmark (CPU only)")
        return {}
    
    results = {}
    
    for size in sizes:
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Get initial memory usage
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            
            # Create tensor
            tensor = torch.randn(size, size, device=device)
            
            # Get memory usage after tensor creation
            torch.cuda.synchronize()
            final_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            
            # Calculate memory used by tensor
            memory_used = final_memory - initial_memory
            
            results[size] = {
                "tensor_shape": f"{size}x{size}",
                "memory_mb": memory_used
            }
            
            # Clean up
            del tensor
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"Error benchmarking size {size}: {e}")
            results[size] = {
                "tensor_shape": f"{size}x{size}",
                "memory_mb": None,
                "error": str(e)
            }
    
    return results

def benchmark_3d_operations(device, batch_sizes=[1, 2, 4, 8]):
    """Benchmark common 3D operations used in model training"""
    if device == "cpu":
        print("Skipping 3D operations benchmark (CPU only)")
        return {}
    
    results = {}
    
    # Define a simple 3D mesh with vertices and faces
    for batch_size in batch_sizes:
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Create vertices and faces
            vertices = torch.randn(batch_size, 1000, 3, device=device)
            faces = torch.randint(0, 1000, (batch_size, 2000, 3), device=device)
            
            # Benchmark vertex transformation (common in 3D generation)
            start_time = time.time()
            for _ in range(10):
                # Apply random rotation
                rotation = torch.randn(batch_size, 3, 3, device=device)
                transformed_vertices = torch.bmm(vertices, rotation)
            torch.cuda.synchronize()
            vertex_transform_time = (time.time() - start_time) / 10
            
            # Benchmark face normal calculation (common in 3D rendering)
            start_time = time.time()
            for _ in range(10):
                # Get vertices for each face
                v0 = torch.gather(vertices, 1, faces[:, :, 0:1].expand(-1, -1, 3))
                v1 = torch.gather(vertices, 1, faces[:, :, 1:2].expand(-1, -1, 3))
                v2 = torch.gather(vertices, 1, faces[:, :, 2:3].expand(-1, -1, 3))
                
                # Calculate face normals
                e1 = v1 - v0
                e2 = v2 - v0
                face_normals = torch.cross(e1, e2, dim=2)
                face_normals = face_normals / (torch.norm(face_normals, dim=2, keepdim=True) + 1e-10)
            torch.cuda.synchronize()
            normal_calc_time = (time.time() - start_time) / 10
            
            results[batch_size] = {
                "batch_size": batch_size,
                "vertex_transform_time_ms": vertex_transform_time * 1000,
                "normal_calc_time_ms": normal_calc_time * 1000
            }
            
            # Clean up
            del vertices, faces, rotation, transformed_vertices
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"Error benchmarking batch size {batch_size}: {e}")
            results[batch_size] = {
                "batch_size": batch_size,
                "vertex_transform_time_ms": None,
                "normal_calc_time_ms": None,
                "error": str(e)
            }
    
    return results

def estimate_training_resources(gpu_info, benchmark_results, dataset_size=1000, epochs=100):
    """Estimate training time and resource requirements"""
    if not gpu_info["cuda_available"]:
        print("Cannot estimate training resources without GPU")
        return {
            "estimated_batch_size": 1,
            "estimated_training_time_hours": "Unknown (CPU only)",
            "memory_required_gb": "Unknown",
            "recommended_settings": {
                "batch_size": 1,
                "gradient_accumulation_steps": 4,
                "mixed_precision": False,
                "note": "Training on CPU is not recommended for this model"
            }
        }
    
    # Get the first GPU info
    gpu = gpu_info["devices"][0]
    total_memory_gb = gpu["total_memory_gb"]
    
    # Estimate batch size based on available memory
    # This is a very rough estimate and depends on the specific model architecture
    if total_memory_gb >= 24:
        estimated_batch_size = 8
    elif total_memory_gb >= 16:
        estimated_batch_size = 4
    elif total_memory_gb >= 12:
        estimated_batch_size = 2
    else:
        estimated_batch_size = 1
    
    # Estimate memory required
    memory_per_sample = 0
    if benchmark_results and estimated_batch_size in benchmark_results:
        # Use benchmark results if available
        vertex_time = benchmark_results[estimated_batch_size]["vertex_transform_time_ms"]
        normal_time = benchmark_results[estimated_batch_size]["normal_calc_time_ms"]
        
        # Rough estimate of time per batch in seconds
        time_per_batch = (vertex_time + normal_time) / 1000 * 10  # Scale factor for full model
    else:
        # Fallback estimates
        if total_memory_gb >= 24:
            time_per_batch = 0.5
        elif total_memory_gb >= 16:
            time_per_batch = 1.0
        elif total_memory_gb >= 12:
            time_per_batch = 2.0
        else:
            time_per_batch = 4.0
    
    # Calculate total training time
    steps_per_epoch = dataset_size / estimated_batch_size
    total_steps = steps_per_epoch * epochs
    total_training_time_hours = (total_steps * time_per_batch) / 3600
    
    # Memory required estimate (very rough)
    memory_required_gb = min(total_memory_gb * 0.8, 2.0 + estimated_batch_size * 1.5)
    
    # Recommended settings
    use_mixed_precision = gpu["compute_capability"] >= "7.0"  # Tensor cores available in Volta+
    
    return {
        "estimated_batch_size": estimated_batch_size,
        "estimated_training_time_hours": total_training_time_hours,
        "memory_required_gb": memory_required_gb,
        "recommended_settings": {
            "batch_size": estimated_batch_size,
            "gradient_accumulation_steps": max(1, 8 // estimated_batch_size),
            "mixed_precision": use_mixed_precision,
            "learning_rate": 1e-5 * (estimated_batch_size / 4)  # Scale LR with batch size
        }
    }

def generate_config_file(output_path, gpu_info, system_info, benchmark_results, training_estimates):
    """Generate a configuration file with optimal settings"""
    config = {
        "hardware": {
            "gpu": gpu_info,
            "system": system_info
        },
        "benchmarks": benchmark_results,
        "training_estimates": training_estimates,
        "recommended_training_args": {
            "batch_size": training_estimates["recommended_settings"]["batch_size"],
            "gradient_accumulation_steps": training_estimates["recommended_settings"]["gradient_accumulation_steps"],
            "learning_rate": training_estimates["recommended_settings"]["learning_rate"],
            "use_amp": training_estimates["recommended_settings"]["mixed_precision"],
            "num_workers": min(system_info["cpu"]["physical_cores"], 8),
            "device": gpu_info["device"]
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {output_path}")
    return config

def main(args):
    """Main function to check and configure GPU resources"""
    print("Checking GPU resources...")
    gpu_info = check_cuda_availability()
    
    if gpu_info["cuda_available"]:
        print(f"Found {gpu_info['device_count']} GPU(s):")
        for device in gpu_info["devices"]:
            print(f"  {device['name']} ({device['total_memory_gb']:.1f} GB)")
    else:
        print("No GPU found. Training will be very slow on CPU.")
    
    print("\nChecking system resources...")
    system_info = check_system_resources()
    print(f"CPU: {system_info['cpu']['physical_cores']} physical cores, {system_info['cpu']['logical_cores']} logical cores")
    print(f"RAM: {system_info['memory']['total_gb']:.1f} GB total, {system_info['memory']['available_gb']:.1f} GB available")
    print(f"Disk: {system_info['disk']['total_gb']:.1f} GB total, {system_info['disk']['free_gb']:.1f} GB free")
    
    # Run benchmarks if a GPU is available
    benchmark_results = {}
    if gpu_info["cuda_available"]:
        print("\nRunning GPU memory benchmark...")
        memory_benchmark = benchmark_gpu_memory(gpu_info["device"])
        
        print("\nRunning 3D operations benchmark...")
        operations_benchmark = benchmark_3d_operations(gpu_info["device"])
        
        benchmark_results = {
            "memory": memory_benchmark,
            "operations": operations_benchmark
        }
    
    # Estimate training resources
    print("\nEstimating training resources...")
    training_estimates = estimate_training_resources(
        gpu_info, 
        benchmark_results.get("operations", {}),
        dataset_size=args.dataset_size,
        epochs=args.epochs
    )
    
    print(f"Estimated batch size: {training_estimates['estimated_batch_size']}")
    print(f"Estimated training time: {training_estimates['estimated_training_time_hours']:.1f} hours")
    print(f"Estimated memory required: {training_estimates['memory_required_gb']:.1f} GB")
    print("\nRecommended settings:")
    for key, value in training_estimates["recommended_settings"].items():
        print(f"  {key}: {value}")
    
    # Generate configuration file
    if args.output:
        config = generate_config_file(
            args.output,
            gpu_info,
            system_info,
            benchmark_results,
            training_estimates
        )
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check and configure GPU resources for Hunyuan3D training")
    parser.add_argument("--output", type=str, default="gpu_config.json",
                        help="Output path for configuration file")
    parser.add_argument("--dataset_size", type=int, default=1000,
                        help="Number of samples in the dataset")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    
    args = parser.parse_args()
    sys.exit(main(args))
