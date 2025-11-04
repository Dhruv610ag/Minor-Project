import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import time
import psutil
import os
from restormer.models.restormer import RestormerTeacher
from  restormer.models.ghostnet import GhostNetFeatureExtractor, SRNetwork, IntegratedGhostSR

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelVisualizer:
    def __init__(self, teacher_model_path, student_model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.teacher_model = self.load_teacher_model(teacher_model_path)
        self.student_model = self.load_student_model(student_model_path)
        
    def load_teacher_model(self, model_path):
        """Load the trained Restormer teacher model"""
        print("📥 Loading Teacher Model...")
        
        model = RestormerTeacher(
            checkpoint_path=model_path,
            scale_factor=4,
            device=self.device
        )
        
        print(f"✅ Teacher Model loaded on {self.device}")
        print(f"📊 Teacher Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    
    def load_student_model(self, model_path):
        """Load the trained GhostNet student model"""
        print("📥 Loading Student Model...")
        
        # Recreate the student model architecture
        input_channels = 9  # 3 frames * 3 channels
        ghostnet_fe = GhostNetFeatureExtractor(in_channels=input_channels)
        sr_net = SRNetwork(in_channels=32, out_channels=3, scale_factor=4)
        model = IntegratedGhostSR(ghostnet_fe, sr_net).to(self.device)
        
        # Load trained weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'student_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['student_state_dict'])
                print("✅ Loaded student state dict from checkpoint")
            else:
                model.load_state_dict(checkpoint)
                print("✅ Loaded student weights directly")
        else:
            print("❌ Student model file not found!")
            
        print(f"✅ Student Model loaded on {self.device}")
        print(f"📊 Student Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    
    def measure_inference_time(self, model, input_size=(3, 256, 256), num_runs=10):
        """Measure inference time for a model"""
        model.eval()
        
        # Create dummy input
        if model == self.teacher_model:
            dummy_input = torch.randn(1, *input_size).to(self.device)
        else:
            # Student expects multiple frames: [B, 9, H, W] for 3 frames
            dummy_input = torch.randn(1, 9, input_size[1], input_size[2]).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
        
        # Measure inference time
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        return np.mean(times), np.std(times)
    
    def measure_memory_usage(self, model, batch_sizes=[1, 2, 4, 8, 16]):
        """Measure memory usage for different batch sizes"""
        memory_usage = []
        
        for batch_size in batch_sizes:
            if model == self.teacher_model:
                dummy_input = torch.randn(batch_size, 3, 256, 256).to(self.device)
            else:
                dummy_input = torch.randn(batch_size, 9, 256, 256).to(self.device)
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            if self.device == 'cuda':
                memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 ** 2)
            
            memory_usage.append(memory_mb)
            
        return batch_sizes, memory_usage
    
    def plot_performance_comparison(self):
        """Create comprehensive performance comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Restormer Teacher vs GhostNet Student - Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Inference Time vs Input Size
        input_sizes = [(3, 128, 128), (3, 256, 256), (3, 512, 512), (3, 1024, 1024)]
        teacher_times = []
        student_times = []
        
        for size in input_sizes:
            teacher_time, _ = self.measure_inference_time(self.teacher_model, size)
            student_time, _ = self.measure_inference_time(self.student_model, size)
            teacher_times.append(teacher_time)
            student_times.append(student_time)
        
        input_pixels = [size[1] * size[2] for size in input_sizes]
        
        axes[0, 0].plot(input_pixels, teacher_times, 'o-', linewidth=2, label='Teacher (Restormer)', markersize=8)
        axes[0, 0].plot(input_pixels, student_times, 's-', linewidth=2, label='Student (GhostNet)', markersize=8)
        axes[0, 0].set_xlabel('Input Image Size (pixels)')
        axes[0, 0].set_ylabel('Inference Time (ms)')
        axes[0, 0].set_title('Inference Time vs Input Size')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Memory Usage vs Batch Size
        batch_sizes, teacher_memory = self.measure_memory_usage(self.teacher_model)
        _, student_memory = self.measure_memory_usage(self.student_model)
        
        axes[0, 1].plot(batch_sizes, teacher_memory, 'o-', linewidth=2, label='Teacher (Restormer)', markersize=8)
        axes[0, 1].plot(batch_sizes, student_memory, 's-', linewidth=2, label='Student (GhostNet)', markersize=8)
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Memory Usage vs Batch Size')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Model Size Comparison
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        
        models = ['Teacher\n(Restormer)', 'Student\n(GhostNet)']
        parameters = [teacher_params / 1e6, student_params / 1e6]  # Convert to millions
        
        bars = axes[1, 0].bar(models, parameters, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[1, 0].set_ylabel('Parameters (Millions)')
        axes[1, 0].set_title('Model Size Comparison')
        
        # Add value labels on bars
        for bar, param in zip(bars, parameters):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{param:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        # 4. Speedup Factor
        speedup_factors = [teacher_times[i] / student_times[i] for i in range(len(teacher_times))]
        
        axes[1, 1].plot(input_pixels, speedup_factors, '^-', linewidth=2, color='#45B7D1', markersize=8)
        axes[1, 1].set_xlabel('Input Image Size (pixels)')
        axes[1, 1].set_ylabel('Speedup Factor (Teacher/Student)')
        axes[1, 1].set_title('Inference Speedup Factor')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add horizontal line at y=1 for reference
        axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'teacher_times': teacher_times,
            'student_times': student_times,
            'teacher_memory': teacher_memory,
            'student_memory': student_memory,
            'teacher_params': teacher_params,
            'student_params': student_params,
            'speedup_factors': speedup_factors
        }
    
    def plot_training_metrics(self, train_logs_path=None):
        """Plot training metrics if logs are available"""
        # If you have training logs, you can load them here
        # For now, creating a template for PSNR/SSIM trends
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Example data - replace with your actual training logs
        epochs = list(range(1, 51))
        
        # Simulated PSNR trends (replace with your actual data)
        teacher_psnr = [28 + 0.1 * i + np.random.normal(0, 0.1) for i in range(50)]
        student_psnr = [26 + 0.15 * i + np.random.normal(0, 0.15) for i in range(50)]
        
        # Simulated SSIM trends (replace with your actual data)
        teacher_ssim = [0.85 + 0.002 * i + np.random.normal(0, 0.005) for i in range(50)]
        student_ssim = [0.82 + 0.003 * i + np.random.normal(0, 0.008) for i in range(50)]
        
        axes[0].plot(epochs, teacher_psnr, 'o-', linewidth=2, label='Teacher PSNR', markersize=4, alpha=0.8)
        axes[0].plot(epochs, student_psnr, 's-', linewidth=2, label='Student PSNR', markersize=4, alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('PSNR Progress During Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, teacher_ssim, 'o-', linewidth=2, label='Teacher SSIM', markersize=4, alpha=0.8)
        axes[1].plot(epochs, student_ssim, 's-', linewidth=2, label='Student SSIM', markersize=4, alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('SSIM')
        axes[1].set_title('SSIM Progress During Training')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_detailed_comparison(self, performance_data):
        """Print detailed performance comparison"""
        print("\n" + "="*60)
        print("📊 DETAILED PERFORMANCE COMPARISON")
        print("="*60)
        
        print(f"\n🧠 MODEL COMPLEXITY:")
        print(f"   Teacher Parameters: {performance_data['teacher_params']:,} ({performance_data['teacher_params']/1e6:.1f}M)")
        print(f"   Student Parameters: {performance_data['student_params']:,} ({performance_data['student_params']/1e6:.1f}M)")
        print(f"   Compression Ratio: {performance_data['teacher_params']/performance_data['student_params']:.2f}x")
        
        print(f"\n⚡ INFERENCE SPEED (256x256 input):")
        print(f"   Teacher Time: {performance_data['teacher_times'][1]:.2f} ms")
        print(f"   Student Time: {performance_data['student_times'][1]:.2f} ms")
        print(f"   Speedup: {performance_data['teacher_times'][1]/performance_data['student_times'][1]:.2f}x")
        
        print(f"\n💾 MEMORY USAGE (Batch Size 4):")
        print(f"   Teacher Memory: {performance_data['teacher_memory'][2]:.1f} MB")
        print(f"   Student Memory: {performance_data['student_memory'][2]:.1f} MB")
        print(f"   Memory Saving: {performance_data['teacher_memory'][2]/performance_data['student_memory'][2]:.2f}x")

# Usage example:
def main():
    # Initialize the visualizer with your model paths
    visualizer = ModelVisualizer(
        teacher_model_path="/kaggle/input/teacher-model/pytorch/default/1/motion_deblurring.pth",  # Update with your path
        student_model_path="/kaggle/input/student-model/pytorch/default/1/final_student_model.pth"   # Update with your path
    )
    
    # Generate performance comparison plots
    performance_data = visualizer.plot_performance_comparison()
    
    # Print detailed comparison
    visualizer.print_detailed_comparison(performance_data)
    
    # Plot training metrics (if you have training logs)
    visualizer.plot_training_metrics()

if __name__ == "__main__":
    main()