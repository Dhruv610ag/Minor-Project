import os
import torch
import numpy as np 
import matplotlib.pyplot as plt  
from torchvision.utils import make_grid  
import time  
from datetime import datetime  

def setup_device(gpu_id=None):
    """Setup the computing device with improved error handling"""
    try:
        if torch.cuda.is_available():
            print("🎮 Available GPUs:")
            for i in range(torch.cuda.device_count()):
                mem_info = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem_info:.1f} GB)")

            if gpu_id is not None:
                if gpu_id < torch.cuda.device_count():
                    device = torch.device(f"cuda:{gpu_id}")
                    torch.cuda.set_device(gpu_id)
                    print(f" Using specified GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                else:
                    print(f"  GPU {gpu_id} not available, using GPU 0")
                    device = torch.device("cuda:0")
            else:
                # Auto-select GPU with most free memory
                free_memories = []
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    free_memories.append(torch.cuda.memory_reserved())
                
                best_gpu = free_memories.index(max(free_memories))
                device = torch.device(f"cuda:{best_gpu}")
                torch.cuda.set_device(best_gpu)
                print(f"Auto-selected GPU {best_gpu} (most free memory)")

            # Print GPU memory info
            print(f" GPU Memory - Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB, "
                  f"Cached: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
            
            torch.cuda.empty_cache()
            
        else:
            device = torch.device("cpu")
            print("⚠  CUDA not available, using CPU")

        return device

    except Exception as e:
        print(f" Error setting up device: {e}")
        return torch.device("cpu")

def create_experiment_name(base_name):
    """Create unique experiment name with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"

def setup_logging(experiment_name, log_dir):
    """Create experiment directories"""
    experiment_dir = os.path.join(log_dir, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return experiment_dir, checkpoint_dir


def check_dataset_structure(data_path):
    """Check and analyze the dataset structure with better reporting"""
    print("\n---  Dataset Structure Analysis ---")

    if not os.path.exists(data_path):
        print(f" ERROR: {data_path} does not exist!")
        return False

    print(f" Contents of {data_path}:")
    data_contents = os.listdir(data_path)
    print(f"Found {len(data_contents)} items: {data_contents[:10]}{'...' if len(data_contents) > 10 else ''}")

    # Check for common Vimeo dataset structures
    possible_paths = [
        os.path.join(data_path, "vimeo_septtuplet", "sequences"),
        os.path.join(data_path, "vimeo_septtuplet"),
        os.path.join(data_path, "sequences"),
        os.path.join(data_path, "sequence"),
        data_path
    ]

    valid_sequence_path = None
    for path in possible_paths:
        if os.path.exists(path):
            valid_sequence_path = path
            print(f" Found valid path: {path}")
            break

    if valid_sequence_path:
        # Count all sequences recursively
        sequence_count = 0
        for root, dirs, files in os.walk(valid_sequence_path):
            if any(f.endswith(('.png', '.jpg', '.jpeg')) for f in files):
                sequence_count += 1
                if sequence_count <= 3:  # Show first 3 sequences
                    print(f"   Sample sequence: {root} -> {[f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))][:3]}")
        
        print(f" Total sequences found: {sequence_count}")

    # Check for split files
    split_files = {
        "sep_trainlist.txt": "Training",
        "sep_testlist.txt": "Testing",
    }

    for file_name, split_type in split_files.items():
        file_path = os.path.join(data_path,"vimeo_settuplet_1", file_name)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                print(f" Found {split_type} list: {file_path} with {len(lines)} entries")
                if lines:
                    print(f"   Sample entries: {lines[:3]}")
            except Exception as e:
                print(f" Error reading {file_path}: {e}")
        else:
            print(f"⚠  {split_type} list not found: {file_path}")