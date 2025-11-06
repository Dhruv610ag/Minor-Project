import base64
import gc
import io
import os
import tempfile
import threading
import time

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image
# Import your models
import sys

from restormer.models.restormer import RestormerTeacher
from restormer.models.ghostnet import GhostNetStudentSR
from restormer.models.sr_network import SRNetwork, IntegratedGhostSR

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max file size

# Global variables
student_model = None
teacher_model = None
processing_status = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_model():
    """Initialize the student model for inference"""
    global student_model, teacher_model
    
    try:
        # Load student model (your trained GhostNet)
        student_checkpoint = "/kaggle/working/checkpoints/final_model.pth"
        
        if os.path.exists(student_checkpoint):
            print("Loading student model...")
            
            # Create student model architecture
            ghostnet = GhostNetStudentSR(scale_factor=4)
            sr_net = SRNetwork(in_channels=32, out_channels=3, num_res_blocks=5, scale_factor=4)
            student_model = IntegratedGhostSR(ghostnet, sr_net).to(device)
            
            # Load checkpoint
            checkpoint = torch.load(student_checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and 'student_state_dict' in checkpoint:
                student_model.load_state_dict(checkpoint['student_state_dict'])
            elif isinstance(checkpoint, dict):
                student_model.load_state_dict(checkpoint)
            else:
                student_model.load_state_dict(checkpoint)
                
            student_model.eval()
            print("✅ Student model loaded successfully!")
        else:
            print(f"❌ Student checkpoint not found at {student_checkpoint}")
            # Create a dummy model for testing
            ghostnet = GhostNetStudentSR(scale_factor=4)
            sr_net = SRNetwork(in_channels=32, out_channels=3, num_res_blocks=5, scale_factor=4)
            student_model = IntegratedGhostSR(ghostnet, sr_net).to(device)
            print("⚠️ Using untrained student model for testing")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Fallback to dummy model
        ghostnet = GhostNetStudentSR(scale_factor=4)
        sr_net = SRNetwork(in_channels=32, out_channels=3, num_res_blocks=5, scale_factor=4)
        student_model = IntegratedGhostSR(ghostnet, sr_net).to(device)
        student_model.eval()
        print("⚠️ Using fallback model")


def image_to_base64(image_array):
    """Convert numpy array to base64"""
    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"


def is_video_file(filename):
    """Check if file is a video"""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
    return any(filename.lower().endswith(ext) for ext in video_extensions)


def extract_frames_from_video(video_path, max_frames=100):
    """Extract frames from video with memory optimization"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame skip to stay within max_frames
    frame_skip = max(1, total_frames // max_frames)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        frame_count += 1
        if len(frames) >= max_frames:
            break
            
    cap.release()
    return frames, fps


def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def preprocess_frame_for_student(frame, scale_factor=4):
    """Prepare frame for student model (3-frame input)"""
    # For student model, we need multiple frames
    # Since we only have one frame, we'll duplicate it to create a 3-frame sequence
    h, w = frame.shape[:2]
    lr_h, lr_w = h // scale_factor, w // scale_factor
    
    # Create LR version
    lr_frame = cv2.resize(frame, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
    
    # Duplicate to create 3-frame sequence
    frame_sequence = [lr_frame, lr_frame, lr_frame]  # Using same frame 3 times
    
    # Convert to tensor [1, 3, 3, H, W] -> [1, 9, H, W]
    frames_tensor = torch.from_numpy(np.stack(frame_sequence)).float() / 255.0
    frames_tensor = frames_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # [1, 3, 3, H, W]
    B, N, C, H, W = frames_tensor.shape
    frames_tensor = frames_tensor.view(B, N * C, H, W)
    
    # Create bicubic upsampled version
    bicubic_frame = cv2.resize(lr_frame, (w, h), interpolation=cv2.INTER_CUBIC)
    bicubic_tensor = torch.from_numpy(bicubic_frame).float() / 255.0
    bicubic_tensor = bicubic_tensor.permute(2, 0, 1).unsqueeze(0)
    
    return frames_tensor.to(device), bicubic_tensor.to(device)


def process_single_frame(frame):
    """Process a single frame with the student model"""
    if student_model is None:
        raise ValueError("Model not loaded")
    
    # Preprocess for student model (3-frame input)
    input_tensor, bicubic_tensor = preprocess_frame_for_student(frame)
    
    with torch.no_grad():
        output_tensor = student_model(input_tensor, bicubic_tensor)
    
    # Convert back to numpy
    output_np = output_tensor.squeeze(0).cpu().numpy()
    output_np = np.transpose(output_np, (1, 2, 0))
    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
    
    return output_np


def process_video_frames(frames, task_id):
    """Process video frames with memory optimization"""
    temp_dir = None
    try:
        processing_status[task_id] = {
            "progress": 0,
            "status": "processing",
            "total": len(frames),
        }
        
        temp_dir = tempfile.mkdtemp()
        frame_paths = []
        
        for i, frame in enumerate(frames):
            try:
                clear_gpu_memory()
                
                # Process frame
                sr_frame = process_single_frame(frame)
                
                # Save processed frame
                frame_filename = f"frame_{i:06d}.png"
                frame_path = os.path.join(temp_dir, frame_filename)
                cv2.imwrite(frame_path, cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR))
                frame_paths.append(frame_path)
                
                # Update progress
                progress = int((i + 1) / len(frames) * 100)
                processing_status[task_id]["progress"] = progress
                
                # Clear memory periodically
                if i % 10 == 0:
                    clear_gpu_memory()
                    
            except Exception as frame_error:
                print(f"Error processing frame {i}: {frame_error}")
                continue
                
        processing_status[task_id]["status"] = "completed"
        processing_status[task_id]["temp_dir"] = temp_dir
        processing_status[task_id]["frame_paths"] = frame_paths
        
    except Exception as e:
        processing_status[task_id]["status"] = "error"
        processing_status[task_id]["error"] = str(e)
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def create_video_from_frames(frame_paths, fps, output_path):
    """Create video from processed frames"""
    if not frame_paths:
        return False
        
    # Get dimensions from first frame
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        return False
        
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
            
    out.release()
    return True


# Flask Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if student_model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
        
    is_video = is_video_file(file.filename)
    
    try:
        if is_video:
            return process_video_upload(file)
        else:
            return process_image_upload(file)
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def process_image_upload(file):
    """Process single image"""
    try:
        clear_gpu_memory()
        
        # Read and decode image
        image_bytes = file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with model
        sr_image = process_single_frame(image)
        
        # Create bicubic comparison
        bicubic_image = cv2.resize(image, (sr_image.shape[1], sr_image.shape[0]), 
                                 interpolation=cv2.INTER_CUBIC)
        
        # Convert to base64
        input_b64 = image_to_base64(image)
        bicubic_b64 = image_to_base64(bicubic_image)
        sr_b64 = image_to_base64(sr_image)
        
        # Calculate simple PSNR
        bicubic_float = bicubic_image.astype(np.float32) / 255.0
        sr_float = sr_image.astype(np.float32) / 255.0
        mse = np.mean((bicubic_float - sr_float) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        return jsonify({
            "success": True,
            "type": "image",
            "input_image": input_b64,
            "bicubic_image": bicubic_b64,
            "sr_image": sr_b64,
            "metrics": {
                "input_size": f"{image.shape[1]}x{image.shape[0]}",
                "output_size": f"{sr_image.shape[1]}x{sr_image.shape[0]}",
                "scale_factor": "4x",
                "psnr_improvement": f"{psnr:.2f} dB",
            }
        })
        
    except Exception as e:
        clear_gpu_memory()
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500


def process_video_upload(file):
    """Process video upload"""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        file.save(tmp_video.name)
        temp_video_path = tmp_video.name
        
    try:
        # Extract frames
        frames, fps = extract_frames_from_video(temp_video_path, max_frames=80)
        
        if len(frames) == 0:
            return jsonify({"error": "Could not extract frames from video"}), 400
            
        task_id = str(int(time.time() * 1000))
        
        # Start background processing
        thread = threading.Thread(target=process_video_frames, args=(frames, task_id))
        thread.start()
        
        return jsonify({
            "success": True,
            "type": "video",
            "task_id": task_id,
            "total_frames": len(frames),
            "fps": fps,
            "message": "Video processing started"
        })
        
    finally:
        os.unlink(temp_video_path)


@app.route("/progress/<task_id>")
def get_progress(task_id):
    """Get processing progress"""
    if task_id not in processing_status:
        return jsonify({"error": "Task not found"}), 404
        
    status = processing_status[task_id].copy()
    
    # Remove large data from response
    for key in ["frames", "temp_dir", "frame_paths"]:
        if key in status:
            del status[key]
            
    return jsonify(status)


@app.route("/result/<task_id>")
def get_result(task_id):
    """Get final video result"""
    if task_id not in processing_status:
        return jsonify({"error": "Task not found"}), 404
        
    status = processing_status[task_id]
    
    if status["status"] != "completed":
        return jsonify({"error": "Processing not completed"}), 400
        
    try:
        frame_paths = status.get("frame_paths", [])
        
        # Create sample preview from middle frame
        if frame_paths:
            middle_idx = len(frame_paths) // 2
            if middle_idx < len(frame_paths):
                # Get original frame (you might want to save original frames too)
                sample_frame_path = frame_paths[middle_idx]
                if os.path.exists(sample_frame_path):
                    sample_frame = cv2.imread(sample_frame_path)
                    sample_frame = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
                    sample_b64 = image_to_base64(sample_frame)
                    
                    return jsonify({
                        "success": True,
                        "sample_sr": sample_b64,
                        "total_frames": len(frame_paths),
                        "message": "Video processing completed"
                    })
                    
        return jsonify({"error": "No frames available"}), 400
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download_video/<task_id>")
def download_video(task_id):
    """Download processed video"""
    if task_id not in processing_status:
        return jsonify({"error": "Task not found"}), 404
        
    status = processing_status[task_id]
    
    if status["status"] != "completed":
        return jsonify({"error": "Processing not completed"}), 400
        
    try:
        frame_paths = status.get("frame_paths", [])
        fps = status.get("fps", 30)
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_output:
            output_path = tmp_output.name
            
        if create_video_from_frames(frame_paths, fps, output_path):
            
            def cleanup_files(temp_dir, output_file):
                try:
                    if temp_dir and os.path.exists(temp_dir):
                        import shutil
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    time.sleep(60)
                    if os.path.exists(output_file):
                        os.unlink(output_file)
                except Exception:
                    pass
                    
            temp_dir = status.get("temp_dir")
            threading.Timer(1, cleanup_files, args=[temp_dir, output_path]).start()
            
            return send_file(
                output_path,
                as_attachment=True,
                download_name=f"super_resolution_output_{task_id}.mp4",
                mimetype="video/mp4"
            )
        else:
            return jsonify({"error": "Failed to create output video"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Initializing Restormer-GhostNet Web Interface...")
    print(f"📱 Using device: {device}")
    init_model()
    print("\n🎯 Starting web server...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("📷 Supports both image and video super-resolution!")
    app.run(debug=True, host="0.0.0.0", port=5000)