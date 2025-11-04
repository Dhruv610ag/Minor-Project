import numpy as np
import torch
import torch.nn.functional as F
from math import exp

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    Args:
        img1, img2: [C, H, W] or [B, C, H, W] tensors in range [0,1]
        max_val: maximum value of the images (default 1.0 for float images)
    Returns:
        PSNR value (float)
    """
    # Ensure both are tensors
    if not isinstance(img1, torch.Tensor):
        img1 = torch.from_numpy(img1)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.from_numpy(img2)
    
    # Ensure same device
    if img1.device != img2.device:
        img2 = img2.to(img1.device)
    
    # Handle batch dimension
    if img1.dim() == 4:
        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        return torch.mean(psnr).item()
    else:
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float("inf")
        return (20 * torch.log10(max_val / torch.sqrt(mse))).item()

def gaussian_kernel(size=11, sigma=1.5):
    """Create 1D Gaussian kernel"""
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g

def create_window(window_size=11, sigma=1.5, channels=3):
    """
    Create a Gaussian window for SSIM calculation.
    """
    _1D_window = gaussian_kernel(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channels, 1, window_size, window_size).contiguous()
    return window

def calculate_ssim(img1, img2, window_size=11, sigma=1.5, size_average=True):
    """
    Calculate the Structural Similarity Index Measure (SSIM) between two images.
    Args:
        img1, img2: [B, C, H, W] or [C, H, W] tensors, in [0,1]
    Returns:
        SSIM value (float)
    """
    # Ensure both are tensors and on same device
    if not isinstance(img1, torch.Tensor):
        img1 = torch.from_numpy(img1)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.from_numpy(img2)
    
    if img1.device != img2.device:
        img2 = img2.to(img1.device)
    
    # Add batch dimension if needed
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    channels = img1.size(1)
    window = create_window(window_size, sigma, channels).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channels) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()

def calculate_motion_consistency(sr_frames, hr_frames):
    """
    Calculate motion consistency score between sequences of frames.
    Args:
        sr_frames: [B, C, H, W] - batch of super-resolved frames (sequential)
        hr_frames: [B, C, H, W] - batch of high-res ground truth frames
    Returns:
        Motion consistency score (float)
    """
    if sr_frames.size(0) < 2:
        return 0.0
    
    # Calculate optical flow-like differences
    sr_diff = torch.mean(torch.abs(sr_frames[1:] - sr_frames[:-1]), dim=[1, 2, 3])
    hr_diff = torch.mean(torch.abs(hr_frames[1:] - hr_frames[:-1]), dim=[1, 2, 3])
    
    # Calculate consistency (lower is better)
    consistency_error = torch.mean(torch.abs(sr_diff - hr_diff))
    
    # Convert to similarity score (higher is better)
    consistency_score = torch.exp(-10.0 * consistency_error).item()
    
    return consistency_score

def calculate_lpips_loss(img1, img2, lpips_model=None):
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity) metric.
    Requires LPIPS model to be passed in.
    """
    if lpips_model is None:
        # Return placeholder if no LPIPS model available
        return 0.0
    
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    # Ensure images are in [-1, 1] range for LPIPS
    img1_lpips = (img1 * 2) - 1
    img2_lpips = (img2 * 2) - 1
    
    return lpips_model(img1_lpips, img2_lpips).mean().item()

def calculate_all_metrics(sr_output, hr_target, lpips_model=None):
    """
    Calculate comprehensive set of metrics for super-resolution evaluation.
    Args:
        sr_output: Super-resolved image(s) [B, C, H, W] or [C, H, W]
        hr_target: High-resolution target image(s) [B, C, H, W] or [C, H, W]
        lpips_model: Optional LPIPS model for perceptual metric
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Basic image quality metrics
    metrics['psnr'] = calculate_psnr(sr_output, hr_target)
    metrics['ssim'] = calculate_ssim(sr_output, hr_target)
    
    # Motion consistency for video sequences
    if (isinstance(sr_output, torch.Tensor) and isinstance(hr_target, torch.Tensor) and
        sr_output.dim() == 4 and hr_target.dim() == 4 and sr_output.size(0) > 1):
        metrics['motion_consistency'] = calculate_motion_consistency(sr_output, hr_target)
    else:
        metrics['motion_consistency'] = 0.0
    
    # Perceptual metric if available
    metrics['lpips'] = calculate_lpips_loss(sr_output, hr_target, lpips_model)
    
    return metrics

def metrics_to_string(metrics, prefix=""):
    """Convert metrics dictionary to readable string"""
    parts = []
    for name, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{prefix}{name}: {value:.4f}")
        else:
            parts.append(f"{prefix}{name}: {value}")
    return ", ".join(parts)