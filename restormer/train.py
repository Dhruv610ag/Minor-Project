import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Import your modules
from dataset import VimeoDataset
from models.restormer import RestormerTeacher
from models.ghostnet import GhostNetStudentSR
from models.mbd import MultiBlockDistillation
from models.feature_alignment import FeatureAlignmentModule
from models.sr_network import SRNetwork, IntegratedGhostSR
from metrices import calculate_all_metrics
from validators import validate_student, validate_teacher, validate_distillation
from utils import setup_device, check_dataset_structure, create_experiment_name, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Restormer-GhostNet Knowledge Distillation Training")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="./data", help="Path to dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Path to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", help="TensorBoard log directory")
    
    # Model parameters
    parser.add_argument("--teacher_checkpoint", type=str, required=True, help="Path to teacher checkpoint")
    parser.add_argument("--scale_factor", type=int, default=4, help="Super-resolution scale factor")
    parser.add_argument("--frame_count", type=int, default=3, help="Number of input frames for student")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
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
    # Teacher model (frozen)
    teacher = RestormerTeacher(
        checkpoint_path=args.teacher_checkpoint,
        scale_factor=args.scale_factor,
        device=device
    )
    
    # Student model
    # Create GhostNet feature extractor and SR reconstruction network, then integrate
    ghostnet = GhostNetStudentSR(scale_factor=args.scale_factor)
    # GhostNet feature extractor outputs 32 channels in the provided implementation
    sr_net = SRNetwork(in_channels=32, out_channels=3, num_res_blocks=5, scale_factor=args.scale_factor)
    student = IntegratedGhostSR(ghostnet, sr_net).to(device)
    
    # Loss functions
    pixel_criterion = nn.L1Loss()  # Pixel-wise loss
    distill_criterion = nn.MSELoss()  # Distillation loss
    
    # Multi-block distillation module
    mbd_module = MultiBlockDistillation(
        distillation_layers=args.distill_layers,
        loss_type=args.distill_loss,
        temperature=1.0
    ).to(device)
    
    # Feature alignment module
    feature_aligner = FeatureAlignmentModule(alignment_type='conv').to(device)
    
    return teacher, student, pixel_criterion, distill_criterion, mbd_module, feature_aligner

def train_epoch(student, teacher, train_loader, pixel_criterion, distill_criterion, 
                mbd_module, optimizer, device, feature_aligner, distill_weight=0.5):
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
            # Teacher processes center frame
            center_frame = lr_frames[:, lr_frames.size(1)//2, :, :, :]
            teacher_output = teacher(center_frame)
        
        # Prepare student input: align/concatenate frames -> [B, N*C, H, W]
        # feature_aligner.forward_student expects frames [B, N, C, H, W]
        lr_input = feature_aligner.forward_student(lr_frames)
        # Student forward expects (x, bicubic)
        student_output = student(lr_input, bicubic_frames)
        
        # Calculate losses
        pixel_loss = pixel_criterion(student_output, hr_frames)
        
        # Distillation loss (simplified - would need feature extraction hooks)
        distill_loss = distill_criterion(student_output, teacher_output.detach())
        
        # Total loss
        loss = pixel_loss + distill_weight * distill_loss
        
        # Backward pass
        loss.backward()
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

def main():
    args = parse_args()
    
    # Setup device
    device = setup_device(args.gpu_id)
    
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
    
    # Optimizer
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Resume training if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"📂 Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            student.load_state_dict(checkpoint['student_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"🔄 Resumed from epoch {start_epoch}")
        else:
            print(f"⚠  No checkpoint found at: {args.resume}")
    
    # Create data loaders
    train_list = os.path.join(args.data_path, "sep_trainlist.txt")
    val_list = os.path.join(args.data_path, "sep_testlist.txt")
    
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
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n📈 Epoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_metrics = train_epoch(
            student, teacher, train_loader, pixel_criterion, distill_criterion,
            mbd_module, optimizer, device, feature_aligner, args.distill_weight
        )
        
        # Validate
        student_metrics = validate_student(student, val_loader, pixel_criterion, device, args.scale_factor)
        teacher_metrics = validate_teacher(teacher, val_loader, device, args.scale_factor)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/Train_Total', train_metrics['total_loss'], epoch)
        writer.add_scalar('Loss/Train_Pixel', train_metrics['pixel_loss'], epoch)
        writer.add_scalar('Loss/Train_Distill', train_metrics['distill_loss'], epoch)
        writer.add_scalar('Loss/Val_Student', student_metrics.get('loss', 0), epoch)
        
        writer.add_scalar('Metrics/PSNR_Student', student_metrics['psnr'], epoch)
        writer.add_scalar('Metrics/PSNR_Teacher', teacher_metrics['psnr'], epoch)
        writer.add_scalar('Metrics/SSIM_Student', student_metrics['ssim'], epoch)
        writer.add_scalar('Metrics/SSIM_Teacher', teacher_metrics['ssim'], epoch)
        
        # Print epoch summary
        print(f"✅ Train Loss: {train_metrics['total_loss']:.4f} "
              f"(Pixel: {train_metrics['pixel_loss']:.4f}, Distill: {train_metrics['distill_loss']:.4f})")
        print(f"📊 Val PSNR - Student: {student_metrics['psnr']:.2f} dB, "
              f"Teacher: {teacher_metrics['psnr']:.2f} dB")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or student_metrics['psnr'] > best_psnr:
            if student_metrics['psnr'] > best_psnr:
                best_psnr = student_metrics['psnr']
                print(f"🎉 New best PSNR: {best_psnr:.2f} dB")
            
            checkpoint = {
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': student_metrics,
                'best_psnr': best_psnr,
                'args': vars(args)
            }
            
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save(student.state_dict(), final_model_path)
    print(f"🏁 Training completed! Final model saved to: {final_model_path}")
    
    writer.close()

if __name__ == "__main__":
    main()