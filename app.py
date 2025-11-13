import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
import time
import io
from math import log10

# Set page config
st.set_page_config(
    page_title="Image Enhancement - Knowledge Distillation",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Enhancement with Knowledge Distillation")
st.markdown("""
Enhance your images using AI models trained with Knowledge Distillation:
- **Teacher Model**: Restormer (Image Enhancement)
- **Student Model**: GhostNet (Image Enhancement)
""")

def calculate_psnr(original, enhanced):
    """Calculate PSNR between original and enhanced images"""
    mse = np.mean((original - enhanced) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, enhanced):
    """Calculate SSIM between original and enhanced images"""
    try:
        from skimage.metrics import structural_similarity as ssim
        # Convert to grayscale for SSIM calculation
        if len(original.shape) == 3:
            original_gray = np.dot(original[...,:3], [0.2989, 0.5870, 0.1140])
            enhanced_gray = np.dot(enhanced[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            original_gray = original
            enhanced_gray = enhanced
        
        # Ensure same size
        min_shape = min(original_gray.shape[0], enhanced_gray.shape[0]), min(original_gray.shape[1], enhanced_gray.shape[1])
        original_gray = original_gray[:min_shape[0], :min_shape[1]]
        enhanced_gray = enhanced_gray[:min_shape[0], :min_shape[1]]
        
        return ssim(original_gray, enhanced_gray, data_range=255)
    except ImportError:
        st.warning("SSIM calculation requires scikit-image. Install with: pip install scikit-image")
        return 0.0

class ImageEnhancer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher = None
        self.student = None
        self.models_loaded = False

    def load_teacher_model(self, model_path):
        """Load Restormer teacher model for image enhancement"""
        try:
            from restormer.models.restormer import RestormerTeacher
            self.teacher = RestormerTeacher(
                checkpoint_path=model_path, 
                scale_factor=1,
                device=self.device
            )
            st.sidebar.success("✅ Teacher model loaded (Image Enhancement)")
            return True
        except Exception as e:
            st.error(f"❌ Error loading teacher model: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False

    def load_student_model(self, model_path):
        """Load the trained student model for image enhancement"""
        try:
            from restormer.models.ghostnet import GhostNetFeatureExtractor
            from restormer.models.sr_network import EnhancementNetwork, IntegratedGhostEnhancer
            ghostnet = GhostNetFeatureExtractor(in_channels=9) 
            enhance_net = EnhancementNetwork(in_channels=32, out_channels=3) 
            self.student = IntegratedGhostEnhancer(ghostnet, enhance_net)
            self.student.to(self.device)
            
            # Load checkpoint
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'student_state_dict' in checkpoint:
                    self.student.load_state_dict(checkpoint['student_state_dict'])
                elif 'model_state_dict' in checkpoint:
                    self.student.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Try direct loading
                    self.student.load_state_dict(checkpoint)
            else:
                st.error(f"❌ Model file not found: {model_path}")
                return False
            
            self.student.eval()
            st.sidebar.success("✅ Student model loaded (Image Enhancement)")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading student model: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False

    def preprocess_for_teacher(self, image):
        """Preprocess image for teacher model (image enhancement)"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size
        
        # Convert to tensor and normalize to [0, 1]
        img_np = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
        
        return img_tensor, original_size, img_np

    def preprocess_for_student(self, image):
        """Preprocess image for student model (image enhancement)"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size
        
        # Convert to tensor and normalize to [0, 1]
        img_np = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
        
        # Student expects 9 channels (3 frames) - repeat the same image
        student_input = img_tensor.repeat(1, 3, 1, 1)  # [1, 9, H, W]
        
        # For enhancement, bicubic is the original image (no upscaling)
        bicubic_tensor = img_tensor.clone()
        
        return student_input, bicubic_tensor, original_size, img_np

    def enhance_image(self, image, model_type='student'):
        """Enhance image using selected model"""
        if not self.models_loaded:
            return None, "Models not loaded", None, None
        
        try:
            original_np = None
            if model_type == 'teacher':
                # Teacher: Image Enhancement
                img_tensor, original_size, original_np = self.preprocess_for_teacher(image)
                img_tensor = img_tensor.to(self.device)
                
                with torch.no_grad():
                    start_time = time.time()
                    output = self.teacher(img_tensor)
                    inference_time = time.time() - start_time
                    output = output.clamp(0, 1)
                
                st.sidebar.write(f"🔍 Teacher output range: [{output.min():.3f}, {output.max():.3f}]")
                
            else:
                # Student: Image Enhancement
                student_input, bicubic_tensor, original_size, original_np = self.preprocess_for_student(image)
                student_input = student_input.to(self.device)
                bicubic_tensor = bicubic_tensor.to(self.device)
                
                with torch.no_grad():
                    start_time = time.time()
                    # Student outputs enhanced image (same resolution)
                    output = self.student(student_input, bicubic_tensor)
                    inference_time = time.time() - start_time
                    output = output.clamp(0, 1)
                
                st.sidebar.write(f"🔍 Student output range: [{output.min():.3f}, {output.max():.3f}]")
                st.sidebar.write(f"🎯 Processing multi-frame enhancement")
            
            # Convert to image
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_np = (output_np * 255).astype(np.uint8)
            enhanced_image = Image.fromarray(output_np)
            
            # Calculate metrics
            psnr_value = None
            ssim_value = None
            
            if original_np is not None:
                # Convert original to same scale for comparison
                original_uint8 = (original_np * 255).astype(np.uint8)
                
                # Ensure same dimensions for metric calculation
                min_height = min(original_uint8.shape[0], output_np.shape[0])
                min_width = min(original_uint8.shape[1], output_np.shape[1])
                
                original_cropped = original_uint8[:min_height, :min_width]
                enhanced_cropped = output_np[:min_height, :min_width]
                
                psnr_value = calculate_psnr(original_cropped, enhanced_cropped)
                ssim_value = calculate_ssim(original_cropped, enhanced_cropped)
            
            return enhanced_image, f"Success - {inference_time:.2f}s", psnr_value, ssim_value
            
        except Exception as e:
            return None, f"Error: {str(e)}", None, None

# Initialize enhancer
if 'enhancer' not in st.session_state:
    st.session_state.enhancer = ImageEnhancer()

enhancer = st.session_state.enhancer

# Sidebar for model loading
st.sidebar.header("🔧 Model Configuration")

# Model paths - UPDATE THESE TO YOUR ACTUAL PATHS
TEACHER_PATH = r"C:\Users\DHRUV AGARWAL\Desktop\Minor-Project\models\TeacherModel\motion_deblurring.pth"
STUDENT_PATH = r"C:\Users\DHRUV AGARWAL\Desktop\Minor-Project\models\StudentModel\final_student_model (1).pth"

# Load models
if st.sidebar.button("🚀 Load All Models", type="primary"):
    with st.spinner("Loading models..."):
        teacher_loaded = enhancer.load_teacher_model(TEACHER_PATH)
        student_loaded = enhancer.load_student_model(STUDENT_PATH)
        
        if teacher_loaded and student_loaded:
            enhancer.models_loaded = True
            st.sidebar.success("✅ All models loaded successfully!")
            
            # Display model info
            if enhancer.teacher:
                teacher_params = sum(p.numel() for p in enhancer.teacher.parameters())
                st.sidebar.info(f"📊 Teacher parameters: {teacher_params:,}")
            if enhancer.student:
                student_params = sum(p.numel() for p in enhancer.student.parameters())
                st.sidebar.info(f"📊 Student parameters: {student_params:,}")
        else:
            st.sidebar.error("❌ Failed to load some models")

# Main interface
st.header("🎨 Image Enhancement")

if enhancer.models_loaded:
    st.success("✅ Models are ready for enhancement!")
else:
    st.warning("⚠️ Please load models first using the button in the sidebar")

# Model selection with clear descriptions
col1, col2 = st.columns(2)
with col1:
    model_type = st.radio(
        "Select Model:",
        ["student", "teacher"],
        index=0,
        help="Choose which model to use for enhancement"
    )

with col2:
    if model_type == "student":
        st.info("""
        **Student Model (GhostNet)**:
        - 🎯 **Task**: Image Enhancement
        - ⚡ **Speed**: Fast inference
        - 🔧 **Method**: Multi-frame processing
        - 📱 **Size**: Lightweight
        - 💡 **Process**: Direct enhancement
        """)
    else:
        st.info("""
        **Teacher Model (Restormer)**:
        - 🎯 **Task**: Image Enhancement
        - 🎨 **Quality**: High quality
        - ⏳ **Speed**: Slower inference
        - 💪 **Power**: Advanced processing
        - 💡 **Process**: Direct enhancement
        """)

# Image upload
uploaded_file = st.file_uploader(
    "Upload an image to enhance",
    type=['png', 'jpg', 'jpeg'],
    help="Supported formats: PNG, JPG, JPEG"
)

if uploaded_file is not None:
    # Display original image
    original_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(original_image, use_container_width=True)
        st.write(f"**Dimensions:** {original_image.size[0]} × {original_image.size[1]}")
        st.write(f"**Mode:** {original_image.mode}")
        
        # FIXED: Consistent task descriptions
        if model_type == "student":
            st.info("🎯 Student will: Process multi-frame input → Enhance details")
        else:
            st.info("🎯 Teacher will: Process single frame → Enhance details")
    
    with col2:
        st.subheader("✨ Enhanced Image")
        
        if enhancer.models_loaded:
            if st.button("🔮 Enhance Image", type="primary", use_container_width=True):
                with st.spinner("Enhancing image..."):
                    enhanced_image, message, psnr_value, ssim_value = enhancer.enhance_image(original_image, model_type)
                    
                    if enhanced_image:
                        st.image(enhanced_image, use_container_width=True)
                        st.write(f"**Dimensions:** {enhanced_image.size[0]} × {enhanced_image.size[1]}")
                        st.success(f"✅ {message}")
                        
                        # Display metrics
                        if psnr_value is not None and ssim_value is not None:
                            col_psnr, col_ssim = st.columns(2)
                            with col_psnr:
                                st.metric(
                                    label="📊 PSNR",
                                    value=f"{psnr_value:.2f} dB",
                                    help="Peak Signal-to-Noise Ratio (higher is better)"
                                )
                            with col_ssim:
                                st.metric(
                                    label="📊 SSIM",
                                    value=f"{ssim_value:.4f}",
                                    help="Structural Similarity Index (1.0 is perfect)"
                                )
                            
                            # Interpretation
                            if psnr_value > 40:
                                st.success("🎉 Excellent quality (PSNR > 40 dB)")
                            elif psnr_value > 30:
                                st.info("👍 Good quality (PSNR 30-40 dB)")
                            else:
                                st.warning("⚠️ Moderate quality (PSNR < 30 dB)")
                            
                            if ssim_value > 0.9:
                                st.success("🎉 Excellent structural similarity (SSIM > 0.9)")
                            elif ssim_value > 0.7:
                                st.info("👍 Good structural similarity (SSIM 0.7-0.9)")
                            else:
                                st.warning("⚠️ Moderate structural similarity (SSIM < 0.7)")
                        
                        # Download button
                        buf = io.BytesIO()
                        enhanced_image.save(buf, format="PNG")
                        st.download_button(
                            label="📥 Download Enhanced Image",
                            data=buf.getvalue(),
                            file_name=f"enhanced_{model_type}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    else:
                        st.error(f"❌ {message}")
        else:
            st.warning("Please load models first!")

# Footer
st.markdown("---")
st.markdown("""
**Metrics Explanation:**
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality (higher = better)
- **SSIM (Structural Similarity Index)**: Measures structural similarity (1.0 = perfect)
""")
# FIXED: Consistent footer
st.markdown(
    "Built with Streamlit | Knowledge Distillation for Image Enhancement | Teacher: Restormer | Student: GhostNet"
)