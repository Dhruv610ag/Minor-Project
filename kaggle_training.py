# %%writefile /kaggle/working/restormer/train.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Import your modules
from restormer.dataset import VimeoDataset
from restormer.models.restormer import RestormerTeacher
from restormer.models.ghostnet import GhostNetFeatureExtractor
from restormer.models.mbd import MultiBlockDistillation
from restormer.models.feature_alignment import FeatureAlignmentModule
from restormer.models.sr_network import SRNetwork, IntegratedGhostSR
from restormer.metrices import calculate_all_metrics, calculate_ssim
from restormer.validators import validate_student, validate_teacher
from restormer.utils import setup_device, check_dataset_structure

# Add missing functions here if not in utils.py
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

def parse_args():
    parser = argparse.ArgumentParser(description="Restormer-GhostNet Knowledge Distillation Training")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="/kaggle/input/archive/vimeo_settuplet_1", help="Path to dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="/kaggle/working/checkpoints", help="Path to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="/kaggle/working/logs", help="TensorBoard log directory")
    
    # Model parameters
    parser.add_argument("--teacher_checkpoint", type=str, required=True, help="Path to teacher checkpoint")
    parser.add_argument("--scale_factor", type=int, default=4, help="Super-resolution scale factor")
    parser.add_argument("--frame_count", type=int, default=3, help="Number of input frames for student")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    
    # Distillation parameters
    parser.add_argument("--distill_weight", type=float, default=0.5, help="Weight for distillation loss")
    parser.add_argument("--distill_layers", nargs='+', default=['encoder1', 'encoder2', 'bottleneck'], 
                       help="Layers for distillation")
    parser.add_argument("--distill_loss", type=str, default='l1', choices=['l1', 'l2', 'kl', 'attention'], 
                       help="Distillation loss type")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    
    return parser.parse_args()

def create_models_and_criteria(args, device):
    """Create teacher, student models and loss criteria"""
    print("🧠 Creating teacher model...")
    # Teacher model (frozen)
    teacher = RestormerTeacher(
        checkpoint_path=args.teacher_checkpoint,
        scale_factor=args.scale_factor,
        device=device
    )
    
    print("🎓 Creating student model...")
    # Student model - use the complete integrated model
    # Calculate input channels: frame_count * 3 (RGB channels)
    input_channels = args.frame_count * 3
    ghostnet_fe = GhostNetFeatureExtractor(in_channels=input_channels)  # Feature extractor
    sr_net = SRNetwork(in_channels=32, out_channels=3, scale_factor=args.scale_factor)  # SR reconstruction
    student = IntegratedGhostSR(ghostnet_fe, sr_net).to(device)
    
    print(f"📊 Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"📊 Student parameters: {sum(p.numel() for p in student.parameters()):,}")
    
    # Loss functions
    pixel_criterion = nn.L1Loss()
    distill_criterion = nn.MSELoss()
    
    # Multi-block distillation module
    mbd_module = MultiBlockDistillation(
        distillation_layers=args.distill_layers,
        loss_type=args.distill_loss,
        temperature=1.0
    ).to(device)
    
    # Feature alignment module
    feature_aligner = FeatureAlignmentModule(
        alignment_type='concat',  # Simple concatenation for frames
        flow_estimation=False
    ).to(device)
    
    return teacher, student, pixel_criterion, distill_criterion, mbd_module, feature_aligner

def train_epoch(student, teacher, train_loader, pixel_criterion, distill_criterion, 
                optimizer, device, feature_aligner, distill_weight=0.5):
    """Train for one epoch"""
    student.train()
    teacher.eval()
    
    total_loss = 0
    total_pixel_loss = 0
    total_distill_loss = 0
    batch_count = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (lr_frames, hr_frames, bicubic_frames) in enumerate(progress_bar):
        # Move data to device
        lr_frames = lr_frames.to(device)
        hr_frames = hr_frames.to(device)
        bicubic_frames = bicubic_frames.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward passes
        with torch.no_grad():
            # Teacher processes center frame only
            center_frame = lr_frames[:, lr_frames.size(1)//2, :, :, :]
            teacher_output = teacher(center_frame)
        
        # Student forward - use feature aligner to prepare input
        # lr_frames shape: [B, N, C, H, W] -> feature_aligner -> [B, N*C, H, W]
        student_input = feature_aligner.forward_student(lr_frames)
        student_output = student(student_input, bicubic_frames)
        
        # Calculate losses
        pixel_loss = pixel_criterion(student_output, hr_frames)
        distill_loss = distill_criterion(student_output, teacher_output.detach())
        
        # Total loss
        loss = pixel_loss + distill_weight * distill_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_pixel_loss += pixel_loss.item()
        total_distill_loss += distill_loss.item()
        batch_count += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Pixel': f'{pixel_loss.item():.4f}',
            'Distill': f'{distill_loss.item():.4f}'
        })
    
    return {
        'total_loss': total_loss / batch_count,
        'pixel_loss': total_pixel_loss / batch_count,
        'distill_loss': total_distill_loss / batch_count
    }

def simple_validate(student, teacher, val_loader, pixel_criterion, device, feature_aligner):
    """Simple validation function with SSIM"""
    student.eval()
    teacher.eval()
    
    total_student_loss = 0
    total_teacher_loss = 0
    student_psnr = 0
    teacher_psnr = 0
    student_ssim = 0
    teacher_ssim = 0
    samples_count = 0
    
    with torch.no_grad():
        for lr_frames, hr_frames, bicubic_frames in val_loader:
            lr_frames = lr_frames.to(device)
            hr_frames = hr_frames.to(device)
            bicubic_frames = bicubic_frames.to(device)
            
            # Teacher inference
            center_frame = lr_frames[:, lr_frames.size(1)//2, :, :, :]
            teacher_output = teacher(center_frame)
            teacher_loss = pixel_criterion(teacher_output, hr_frames)
            
            # Student inference
            student_input = feature_aligner.forward_student(lr_frames)
            student_output = student(student_input, bicubic_frames)
            student_loss = pixel_criterion(student_output, hr_frames)
            
            # Calculate PSNR and SSIM
            batch_size = lr_frames.size(0)
            for i in range(batch_size):
                student_psnr += calculate_all_metrics(student_output[i], hr_frames[i])['psnr']
                teacher_psnr += calculate_all_metrics(teacher_output[i], hr_frames[i])['psnr']
                
                # Calculate SSIM
                student_ssim += calculate_ssim(student_output[i].unsqueeze(0), hr_frames[i].unsqueeze(0))
                teacher_ssim += calculate_ssim(teacher_output[i].unsqueeze(0), hr_frames[i].unsqueeze(0))
            
            total_student_loss += student_loss.item()
            total_teacher_loss += teacher_loss.item()
            samples_count += batch_size
    
    return {
        'student_loss': total_student_loss / len(val_loader),
        'teacher_loss': total_teacher_loss / len(val_loader),
        'student_psnr': student_psnr / samples_count,
        'teacher_psnr': teacher_psnr / samples_count,
        'student_ssim': student_ssim / samples_count,
        'teacher_ssim': teacher_ssim / samples_count
    }

def main():
    args = parse_args()
    
    # Setup device
    device = setup_device(args.gpu_id)
    print(f"🔧 Using device: {device}")
    
    # Create experiment name and directories
    experiment_name = create_experiment_name("restormer_ghostnet_distill")
    experiment_dir, checkpoint_dir = setup_logging(experiment_name, args.log_dir)
    
    print(f"🚀 Starting experiment: {experiment_name}")
    print(f"📁 Experiment directory: {experiment_dir}")
    print(f"💾 Checkpoint directory: {checkpoint_dir}")
    
    # Check dataset structure
    check_dataset_structure(args.data_path)
    
    # Create models and criteria
    teacher, student, pixel_criterion, distill_criterion, mbd_module, feature_aligner = \
        create_models_and_criteria(args, device)
    
    # Optimizer (only for student)
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Create data loaders
    train_list = os.path.join(args.data_path, "sep_trainlist.txt")
    val_list = os.path.join(args.data_path, "sep_testlist.txt")
    
    print("📊 Creating datasets...")
    train_dataset = VimeoDataset(
        args.data_path,
        split_list=train_list if os.path.exists(train_list) else None,
        scale_factor=args.scale_factor,
        frame_count=args.frame_count,
        is_training=True,
        patch_size=64
    )
    
    val_dataset = VimeoDataset(
        args.data_path,
        split_list=val_list if os.path.exists(val_list) else None,
        scale_factor=args.scale_factor,
        frame_count=args.frame_count,
        is_training=False,
        patch_size=128
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"📊 Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, experiment_name))
    
    # Training loop
    best_psnr = 0
    print("\n🎯 Starting training...")
    
    for epoch in range(args.num_epochs):
        print(f"\n📈 Epoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch(
            student, teacher, train_loader, pixel_criterion, distill_criterion,
            optimizer, device, feature_aligner, args.distill_weight
        )
        
        # Validate
        val_metrics = simple_validate(student, teacher, val_loader, pixel_criterion, device, feature_aligner)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/Train_Total', train_metrics['total_loss'], epoch)
        writer.add_scalar('Loss/Train_Pixel', train_metrics['pixel_loss'], epoch)
        writer.add_scalar('Loss/Train_Distill', train_metrics['distill_loss'], epoch)
        writer.add_scalar('Loss/Val_Student', val_metrics['student_loss'], epoch)
        writer.add_scalar('Loss/Val_Teacher', val_metrics['teacher_loss'], epoch)
        
        writer.add_scalar('Metrics/PSNR_Student', val_metrics['student_psnr'], epoch)
        writer.add_scalar('Metrics/PSNR_Teacher', val_metrics['teacher_psnr'], epoch)
        writer.add_scalar('Metrics/SSIM_Student', val_metrics['student_ssim'], epoch)
        writer.add_scalar('Metrics/SSIM_Teacher', val_metrics['teacher_ssim'], epoch)
        
        # Print epoch summary
        print(f"✅ Train Loss: {train_metrics['total_loss']:.4f} "
              f"(Pixel: {train_metrics['pixel_loss']:.4f}, Distill: {train_metrics['distill_loss']:.4f})")
        print(f"📊 Val PSNR - Student: {val_metrics['student_psnr']:.2f} dB, "
              f"Teacher: {val_metrics['teacher_psnr']:.2f} dB")
        print(f"📊 Val SSIM - Student: {val_metrics['student_ssim']:.4f}, "
              f"Teacher: {val_metrics['teacher_ssim']:.4f}")
        print(f"📊 Val Loss - Student: {val_metrics['student_loss']:.4f}, "
              f"Teacher: {val_metrics['teacher_loss']:.4f}")
        
        # Save checkpoint - ONLY every 5 epochs OR when PSNR improves
        should_save_checkpoint = False
        save_reason = ""
        
        if val_metrics['student_psnr'] > best_psnr:
            best_psnr = val_metrics['student_psnr']
            should_save_checkpoint = True
            save_reason = f"🎉 New best PSNR: {best_psnr:.2f} dB"
        
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            should_save_checkpoint = True
            save_reason = f"💾 Regular save at epoch {epoch + 1}"
        
        if should_save_checkpoint:
            print(save_reason)
            
            checkpoint = {
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_psnr': best_psnr,
                'args': vars(args)
            }
            
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # Save final model (always save at the end)
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save({
        'student_state_dict': student.state_dict(),
        'best_psnr': best_psnr,
        'args': vars(args)
    }, final_model_path)
    print(f"🏁 Training completed! Final model saved to: {final_model_path}")
    
    # Print final results
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Best Student PSNR: {best_psnr:.2f} dB")
    print(f"   Final Student PSNR: {val_metrics['student_psnr']:.2f} dB")
    print(f"   Final Student SSIM: {val_metrics['student_ssim']:.4f}")
    print(f"   Teacher PSNR: {val_metrics['teacher_psnr']:.2f} dB")
    print(f"   Teacher SSIM: {val_metrics['teacher_ssim']:.4f}")
    
    writer.close()
    print("✨ Experiment completed successfully!")

if __name__ == "__main__":
    main()