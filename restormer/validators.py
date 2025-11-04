import torch
import torch.nn.functional as F
from restormer.metrices import calculate_all_metrics  # Make sure this imports the updated metrics  # noqa: F401

def validate_teacher(teacher_model, dataloader, device, scale_factor=4):
    """Validate the teacher model (Restormer) on validation dataset"""
    teacher_model.eval()
    metrics = {"psnr": 0.0, "ssim": 0.0, "moc": 0.0}
    samples_count = 0

    with torch.no_grad():
        for lr_frames, hr_frames, bicubic_frames in dataloader:
            # Teacher processes center frame only
            center_frame = lr_frames[:, lr_frames.size(1)//2, :, :, :]  # [B, C, H, W]
            center_frame = center_frame.to(device)
            hr_frames = hr_frames.to(device)

            # Teacher forward pass
            sr_output = teacher_model(center_frame)

            # Calculate metrics for each sample in batch
            for i in range(sr_output.size(0)):
                batch_metrics = calculate_all_metrics(sr_output[i], hr_frames[i])
                for k, v in batch_metrics.items():
                    metrics[k] += v

            samples_count += sr_output.size(0)

    # Average metrics
    for k in metrics:
        metrics[k] /= samples_count

    return metrics

def validate_student(student_model, dataloader, criterion, device, scale_factor=4):
    """Validate the student model (GhostNet) on validation dataset"""
    student_model.eval()
    total_loss = 0
    metrics = {"psnr": 0.0, "ssim": 0.0, "moc": 0.0, "lpips": 0.0}
    samples_count = 0

    with torch.no_grad():
        for lr_frames, hr_frames, bicubic_frames in dataloader:
            lr_frames = lr_frames.to(device)
            hr_frames = hr_frames.to(device)
            bicubic_frames = bicubic_frames.to(device)

            # Student forward pass
            # Convert lr_frames [B, N, C, H, W] -> [B, N*C, H, W] for integrated student
            B, N, C, H, W = lr_frames.shape
            lr_input = lr_frames.view(B, N * C, H, W)
            sr_output = student_model(lr_input, bicubic_frames)

            # Calculate loss
            loss = criterion(sr_output, hr_frames)
            total_loss += loss.item()

            # Calculate metrics for each sample in batch
            for i in range(sr_output.size(0)):
                batch_metrics = calculate_all_metrics(sr_output[i], hr_frames[i])
                for k, v in batch_metrics.items():
                    metrics[k] += v

            samples_count += sr_output.size(0)

    # Average metrics
    for k in metrics:
        metrics[k] /= samples_count
    metrics["loss"] = total_loss / len(dataloader)

    return metrics

def validate_distillation(student_model, teacher_model, dataloader, distillation_criterion, 
                         device, scale_factor=4, mbd_module=None):
    """
    Validate knowledge distillation performance
    Returns both student metrics and distillation metrics
    """
    student_model.eval()
    teacher_model.eval()
    
    metrics = {"psnr": 0.0, "ssim": 0.0, "moc": 0.0, "lpips": 0.0}
    distill_metrics = {"distill_loss": 0.0}
    samples_count = 0

    with torch.no_grad():
        for lr_frames, hr_frames, bicubic_frames in dataloader:
            lr_frames = lr_frames.to(device)
            hr_frames = hr_frames.to(device)
            bicubic_frames = bicubic_frames.to(device)

            # Get center frame for teacher
            center_frame = lr_frames[:, lr_frames.size(1)//2, :, :, :]
            
            # Forward passes
            teacher_output = teacher_model(center_frame)
            # Prepare student input similarly
            B, N, C, H, W = lr_frames.shape
            lr_input = lr_frames.view(B, N * C, H, W)
            student_output = student_model(lr_input, bicubic_frames)

            # Calculate distillation loss if MBD module is provided
            if mbd_module is not None:
                # This would require feature extraction hooks - simplified here
                distill_loss = distillation_criterion(student_output, teacher_output)
                distill_metrics["distill_loss"] += distill_loss.item()

            # Calculate quality metrics
            for i in range(student_output.size(0)):
                batch_metrics = calculate_all_metrics(student_output[i], hr_frames[i])
                for k, v in batch_metrics.items():
                    metrics[k] += v

            samples_count += student_output.size(0)

    # Average metrics
    for k in metrics:
        metrics[k] /= samples_count
    for k in distill_metrics:
        distill_metrics[k] /= len(dataloader)

    return metrics, distill_metrics

def progressive_validation(model, dataloader, device, scale_factors=[2, 3, 4]):
    """
    Validate model performance at different scale factors
    """
    results = {}
    
    for scale in scale_factors:
        print(f"Validating at scale factor {scale}...")
        # This would require a model that supports different scale factors
        # For now, we'll just demonstrate the structure
        metrics = validate_student(model, dataloader, torch.nn.L1Loss(), device, scale)
        results[f"scale_{scale}"] = metrics
    
    return results

def generate_validation_report(student_metrics, teacher_metrics=None, distill_metrics=None):
    """Generate a comprehensive validation report"""
    report = []
    report.append("📊 Validation Report")
    report.append("=" * 50)
    
    if teacher_metrics:
        report.append("\n🧠 Teacher Model (Restormer):")
        for metric, value in teacher_metrics.items():
            report.append(f"  {metric.upper()}: {value:.4f}")
    
    report.append("\n🎓 Student Model (GhostNet):")
    for metric, value in student_metrics.items():
        report.append(f"  {metric.upper()}: {value:.4f}")
    
    if distill_metrics:
        report.append("\n🔗 Distillation Metrics:")
        for metric, value in distill_metrics.items():
            report.append(f"  {metric}: {value:.4f}")
    
    # Calculate improvement over bicubic baseline
    if teacher_metrics and 'psnr' in teacher_metrics and 'psnr' in student_metrics:
        improvement = student_metrics['psnr'] - teacher_metrics['psnr']
        report.append(f"\n📈 PSNR Improvement: {improvement:+.3f} dB")
    
    report.append("=" * 50)
    
    return "\n".join(report)