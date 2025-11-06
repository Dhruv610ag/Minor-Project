import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
import os
import time
import torch.nn.functional as F

# Import your models
import sys
sys.path.append('/kaggle/working/restormer')

from restormer.models.restormer import RestormerTeacher   
from restormer.models.ghostnet import GhostNetStudentSR  
from restormer.metrices import calculate_psnr, calculate_ssim


def setup_device(gpu_id=0):
    """Setup device with proper error handling"""
    if torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            device = torch.device("cuda:0")
            print(f"GPU {gpu_id} not available, using GPU 0")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    return device


def load_model(model_type, checkpoint_path, device, scale_factor=4):
    """Load appropriate model based on type"""
    if model_type == "teacher":
        model = RestormerTeacher(
            checkpoint_path=checkpoint_path,
            scale_factor=scale_factor,
            device=device
        )
    elif model_type == "student":
        # Load student model directly (as used in training)
        model = GhostNetStudentSR(scale_factor=scale_factor).to(device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'student_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['student_state_dict'])
                    elif 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Assume it's the model itself
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)
                    
                print(f"✅ Student model loaded from {checkpoint_path}")
            except Exception as e:
                print(f"❌ Error loading student model: {e}")
                print("⚠️ Using untrained student model")
        else:
            print("⚠️ No checkpoint provided, using untrained student model")
            
        model.eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def prepare_student_input(frame, scale_factor=4):
    """Prepare input for student model (3-frame sequence)"""
    # Convert to RGB and normalize
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
        
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    
    # Create low-resolution version
    h, w = frame_normalized.shape[:2]
    lr_h, lr_w = h // scale_factor, w // scale_factor
    lr_frame = cv2.resize(frame_normalized, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
    
    # Create 3-frame sequence by duplicating (for single frame inference)
    # In real usage, you'd use actual consecutive frames
    frame_sequence = [lr_frame, lr_frame, lr_frame]
    
    # Convert to tensor [1, 3, 3, H, W] -> [1, 9, H, W]
    frames_tensor = torch.from_numpy(np.stack(frame_sequence)).float()
    frames_tensor = frames_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # [1, 3, 3, H, W]
    B, N, C, H, W = frames_tensor.shape
    frames_tensor = frames_tensor.view(B, N * C, H, W)
    
    # Create bicubic upsampled version
    bicubic_frame = cv2.resize(lr_frame, (w, h), interpolation=cv2.INTER_CUBIC)
    bicubic_tensor = torch.from_numpy(bicubic_frame).float()
    bicubic_tensor = bicubic_tensor.permute(2, 0, 1).unsqueeze(0)
    
    return frames_tensor, bicubic_tensor


def process_video(model, input_path, output_path, device, scale_factor=4, model_type="student"):
    """Process a video file using the specified model"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_width, out_height = width * scale_factor, height * scale_factor
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    frame_count = 0
    frame_buffer = []

    with tqdm(total=total_frames, desc=f"Processing video ({model_type})") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if model_type == "teacher":
                # Teacher processes single frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(frame_tensor)

            elif model_type == "student":
                # Student needs 3-frame sequence
                frame_buffer.append(frame)
                if len(frame_buffer) < 3:
                    pbar.update(1)
                    continue
                    
                if len(frame_buffer) > 3:
                    frame_buffer = frame_buffer[-3:]
                
                # Process the middle frame using 3-frame context
                current_frame = frame_buffer[1]  # Use middle frame
                input_tensor, bicubic_tensor = prepare_student_input(current_frame, scale_factor)
                input_tensor = input_tensor.to(device)
                bicubic_tensor = bicubic_tensor.to(device)
                
                with torch.no_grad():
                    output = model(input_tensor, bicubic_tensor)

            # Convert output to numpy
            output_np = output.squeeze(0).cpu().numpy()
            output_np = np.transpose(output_np, (1, 2, 0))
            output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)

            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
            out.write(output_bgr)

            frame_count += 1
            pbar.update(1)

    cap.release()
    out.release()
    print(f"✅ Processed {frame_count} frames. Output saved to: {output_path}")


def process_single_image(model, input_path, output_path, device, scale_factor=4, model_type="student"):
    """Process a single image"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    # Read image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Cannot read image: {input_path}")
    
    if model_type == "teacher":
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            
    elif model_type == "student":
        input_tensor, bicubic_tensor = prepare_student_input(image, scale_factor)
        input_tensor = input_tensor.to(device)
        bicubic_tensor = bicubic_tensor.to(device)
        
        with torch.no_grad():
            output = model(input_tensor, bicubic_tensor)
    
    # Convert output
    output_np = output.squeeze(0).cpu().numpy()
    output_np = np.transpose(output_np, (1, 2, 0))
    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, output_bgr)
    print(f"✅ Image processed and saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Restormer-GhostNet Inference")
    parser.add_argument("--model-type", choices=["teacher", "student"], required=True, 
                       help="Model type to use")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, 
                       help="Input image/video path")
    parser.add_argument("--output", type=str, required=True, 
                       help="Output path")
    parser.add_argument("--scale", type=int, default=4, 
                       help="Scale factor (default: 4)")
    parser.add_argument("--gpu", type=int, default=0, 
                       help="GPU ID (default: 0)")
    
    args = parser.parse_args()

    # Setup device
    device = setup_device(args.gpu)
    
    # Load model
    print(f"Loading {args.model_type} model from {args.checkpoint}...")
    model = load_model(args.model_type, args.checkpoint, device, args.scale)
    print("✅ Model loaded successfully!")

    # Check if input is image or video
    is_video = args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
    
    start_time = time.time()
    
    if is_video:
        print("🎥 Processing video...")
        process_video(model, args.input, args.output, device, args.scale, args.model_type)
    else:
        print("🖼️ Processing image...")
        process_single_image(model, args.input, args.output, device, args.scale, args.model_type)
    
    process_time = time.time() - start_time
    print(f"⏱️ Processing completed in {process_time:.2f} seconds")


if __name__ == "__main__":
    main()