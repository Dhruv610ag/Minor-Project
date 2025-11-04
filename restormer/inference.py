import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
import os
import time
import torch.nn.functional as F
from restormer.models.restormer import RestormerTeacher   
from restormer.models.ghostnet import GhostNetStudentSR  
from restormer.models.sr_network import SRNetwork, IntegratedGhostSR
from restormer.metrices import calculate_psnr, calculate_ssim, calculate_motion_consistency


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


def load_model(model_type, checkpoint_path, device, scale_factor=1):
    """Load appropriate model based on type"""
    if model_type == "teacher":
        model = RestormerTeacher(checkpoint_path=checkpoint_path,
                                 scale_factor=scale_factor,
                                 device=device)
    elif model_type == "student":
        # Instantiate integrated student: GhostNet feature extractor + SR network
        ghostnet = GhostNetStudentSR(scale_factor=scale_factor)
        sr_net = SRNetwork(in_channels=32, out_channels=3, num_res_blocks=5, scale_factor=scale_factor)
        model = IntegratedGhostSR(ghostnet, sr_net).to(device)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # checkpoint may be a dict with various keys
            if isinstance(checkpoint, dict):
                # try common keys
                if 'student_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['student_state_dict'])
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    # assume checkpoint is a state_dict
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
        model.eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def process_video(model, input_path, output_path, device, scale_factor=4, model_type="teacher"):
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

    frame_buffer = []
    frame_count = 0

    with tqdm(total=total_frames, desc=f"Processing video ({model_type})") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            if model_type == "teacher":
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(frame_tensor)

            elif model_type == "student":
                # Build LR buffer by downsampling input frames to LR
                lr_frame = cv2.resize(frame_rgb, (width // scale_factor, height // scale_factor), interpolation=cv2.INTER_AREA)
                frame_buffer.append(lr_frame)
                if len(frame_buffer) < 3:
                    pbar.update(1)
                    continue
                if len(frame_buffer) > 3:
                    frame_buffer = frame_buffer[-3:]
                # stack LR frames [N, H, W, C]
                frame_sequence = np.stack(frame_buffer, axis=0)
                # to tensor [1, N, C, H, W]
                frame_tensor = torch.from_numpy(frame_sequence).permute(0, 3, 1, 2).unsqueeze(0)
                # reshape to concatenated channels [1, N*C, H, W]
                B, N, C, H_lr, W_lr = frame_tensor.shape
                frame_tensor = frame_tensor.view(B, N * C, H_lr, W_lr).to(device)

                # create bicubic upsampled center LR frame to HR size
                center_lr = frame_buffer[len(frame_buffer)//2]
                bicubic_np = cv2.resize(center_lr, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
                bicubic_tensor = torch.from_numpy(bicubic_np).permute(2, 0, 1).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(frame_tensor, bicubic_tensor)

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
    print(f"Processed {frame_count} frames. Output saved to: {output_path}")


def evaluate_video(model, input_path, gt_path, device, scale_factor=4, model_type="teacher"):
    """Evaluate model performance on video with ground truth"""
    cap_input = cv2.VideoCapture(input_path)
    cap_gt = cv2.VideoCapture(gt_path)
    
    if not cap_input.isOpened() or not cap_gt.isOpened():
        raise ValueError("Cannot open input or ground truth video")
    
    total_frames = min(
        int(cap_input.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

    metrics = {"psnr": 0.0, "ssim": 0.0, "moc": 0.0}
    hr_frames, sr_frames, frame_buffer = [], [], []

    with tqdm(total=total_frames, desc="Evaluating video") as pbar:
        for _ in range(total_frames):
            ret_input, frame_input = cap_input.read()
            ret_gt, frame_gt = cap_gt.read()
            if not ret_input or not ret_gt:
                break

            frame_input_rgb = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frame_gt_rgb = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            if model_type == "teacher":
                frame_tensor = torch.from_numpy(frame_input_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            else:  # student
                # downsample to LR and buffer
                lr_frame = cv2.resize(frame_input_rgb, (int(frame_input.shape[1]//scale_factor), int(frame_input.shape[0]//scale_factor)), interpolation=cv2.INTER_AREA)
                frame_buffer.append(lr_frame)
                if len(frame_buffer) < 3:
                    pbar.update(1)
                    continue
                if len(frame_buffer) > 3:
                    frame_buffer = frame_buffer[-3:]
                frame_sequence = np.stack(frame_buffer, axis=0)
                frame_tensor = torch.from_numpy(frame_sequence).permute(0, 3, 1, 2).unsqueeze(0)
                B, N, C, H_lr, W_lr = frame_tensor.shape
                frame_tensor = frame_tensor.view(B, N * C, H_lr, W_lr).to(device)

            # For student, pass bicubic upsampled center LR as second arg; teacher ignores second arg
            with torch.no_grad():
                if model_type == "student":
                    # create bicubic upsampled center LR to match GT/HR size
                    center_lr = frame_buffer[len(frame_buffer)//2]
                    bicubic_np = cv2.resize(center_lr, (frame_gt_rgb.shape[1], frame_gt_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
                    bicubic_tensor = torch.from_numpy(bicubic_np).permute(2, 0, 1).unsqueeze(0).to(device)
                    output = model(frame_tensor, bicubic_tensor)
                else:
                    output = model(frame_tensor)

            output_np = output.squeeze(0).cpu().numpy()
            sr_tensor = torch.from_numpy(output_np)
            hr_tensor = torch.from_numpy(frame_gt_rgb).permute(2, 0, 1)

            # Match sizes if needed
            if sr_tensor.shape[1:] != hr_tensor.shape[1:]:
                hr_tensor = F.interpolate(hr_tensor.unsqueeze(0),
                                          size=sr_tensor.shape[1:],
                                          mode="bicubic",
                                          align_corners=False).squeeze(0)

            sr_frames.append(sr_tensor)
            hr_frames.append(hr_tensor)
            pbar.update(1)

    cap_input.release()
    cap_gt.release()

    if sr_frames and hr_frames:
        sr_tensor = torch.stack(sr_frames)
        hr_tensor = torch.stack(hr_frames)

        for i in range(len(hr_frames)):
            metrics["psnr"] += calculate_psnr(sr_tensor[i], hr_tensor[i])
            metrics["ssim"] += calculate_ssim(sr_tensor[i], hr_tensor[i])

        metrics["psnr"] /= len(hr_frames)
        metrics["ssim"] /= len(hr_frames)
        metrics["moc"] = calculate_motion_consistency(sr_tensor, hr_tensor)

    return metrics, len(hr_frames)


def main():
    parser = argparse.ArgumentParser(description="Video Enhancement Inference")
    parser.add_argument("--model-type", choices=["teacher", "student"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, help="Output video path")
    parser.add_argument("--gt", type=str, help="Ground truth video path for evaluation")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation if ground truth is provided")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor (default: 4)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (default: 0)")
    
    args = parser.parse_args()

    device = setup_device(args.gpu)
    print(f"Loading {args.model_type} model from {args.checkpoint}...")
    model = load_model(args.model_type, args.checkpoint, device, args.scale)
    print("Model loaded successfully!")

    if args.evaluate and args.gt:
        print(f"Evaluating {args.model_type} model...")
        start_time = time.time()
        metrics, processed_frames = evaluate_video(model, args.input, args.gt, device, args.scale, args.model_type)
        eval_time = time.time() - start_time

        print("\n📊 Evaluation Results:")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
        print(f"Motion Consistency: {metrics['moc']:.4f}")
        print(f"Evaluation time: {eval_time:.2f} seconds")
        print(f"FPS: {processed_frames / eval_time:.2f}")

    if args.output:
        print(f"Processing video with {args.model_type} model...")
        start_time = time.time()
        process_video(model, args.input, args.output, device, args.scale, args.model_type)
        process_time = time.time() - start_time
        print(f"Processing completed in {process_time:.2f} seconds")


if __name__ == "__main__":
    main()