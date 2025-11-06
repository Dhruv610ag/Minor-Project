import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
import time
import io

class ImageEnhancementApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher_model = None
        self.student_model = None
        self.models_loaded = False
        
    def load_models(self, teacher_path, student_path):
        """Load both teacher and student models"""
        try:
            # Load teacher model
            if teacher_path and os.path.exists(teacher_path):
                from restormer.models.restormer import RestormerTeacher
                self.teacher_model = RestormerTeacher(
                    checkpoint_path=teacher_path, 
                    device=self.device
                )
                st.success(f"✅ Teacher model loaded ({sum(p.numel() for p in self.teacher_model.parameters()):,} parameters)")
            else:
                st.warning("⚠️ Teacher model path not provided or file doesn't exist")
            
            # Load student model  
            if student_path and os.path.exists(student_path):
                from restormer.models.ghostnet import GhostNetStudentSR
                from restormer.models.sr_network import SRNetwork, IntegratedGhostSR
                
                ghostnet = GhostNetStudentSR()
                sr_net = SRNetwork()
                self.student_model = IntegratedGhostSR(ghostnet, sr_net)
                
                # Load student weights
                student_weights = torch.load(student_path, map_location=self.device)
                self.student_model.load_state_dict(student_weights)
                self.student_model.to(self.device)
                self.student_model.eval()
                
                st.success(f"✅ Student model loaded ({sum(p.numel() for p in self.student_model.parameters()):,} parameters)")
                self.models_loaded = True
            else:
                st.warning("⚠️ Student model path not provided or file doesn't exist")
                
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            self.models_loaded = False

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Store original size
        original_size = image.size
        
        # Convert to tensor and normalize
        img_np = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
        
        return img_tensor, original_size

    def postprocess_image(self, tensor, original_size):
        """Convert model output back to PIL Image"""
        # Convert tensor to numpy
        sr_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sr_np = np.clip(sr_np * 255, 0, 255).astype(np.uint8)
        
        # Create PIL Image
        sr_img = Image.fromarray(sr_np)
        
        return sr_img

    def enhance_image(self, image, model_type='student'):
        """Enhance image using selected model"""
        if not self.models_loaded:
            st.error("Models not loaded! Please load models first.")
            return None
        
        try:
            # Preprocess
            img_tensor, original_size = self.preprocess_image(image)
            img_tensor = img_tensor.to(self.device)
            
            # Select model
            if model_type == 'teacher' and self.teacher_model:
                model = self.teacher_model
            elif model_type == 'student' and self.student_model:
                model = self.student_model
            else:
                st.error(f"Selected model ({model_type}) not available")
                return None
            
            # Inference
            with torch.no_grad():
                start_time = time.time()
                enhanced_tensor = model(img_tensor).clamp(0, 1)
                inference_time = time.time() - start_time
            
            # Postprocess
            enhanced_image = self.postprocess_image(enhanced_tensor, original_size)
            
            st.info(f"🕒 Inference time: {inference_time:.2f} seconds")
            
            return enhanced_image
            
        except Exception as e:
            st.error(f"❌ Error during enhancement: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="Image Enhancement Demo",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🖼️ Image Enhancement with Knowledge Distillation")
    st.markdown("""
    Enhance your images using AI models trained with Knowledge Distillation:
    - **Teacher Model**: Restormer (High quality, slower) - Motion Deblurring
    - **Student Model**: GhostNet (Good quality, faster) - Lightweight version
    """)
    
    # Initialize app
    if 'app' not in st.session_state:
        st.session_state.app = ImageEnhancementApp()
    
    app = st.session_state.app
    
    # Sidebar for model loading
    st.sidebar.header("🔧 Model Configuration")
    
    # Pre-configured model paths
    TEACHER_PATH = r"C:\Users\DHRUV AGARWAL\Desktop\Minor-Project\models\TeacherModel\motion_deblurring.pth"
    STUDENT_PATH = r"C:\Users\DHRUV AGARWAL\Desktop\Minor-Project\models\StudentModel\final_student_model.pth"
    
    with st.sidebar.expander("📁 Load Models", expanded=True):
        st.info(f"**Teacher Model**: `{TEACHER_PATH}`")
        st.info(f"**Student Model**: `{STUDENT_PATH}`")
        
        if st.button("🚀 Load Models"):
            with st.spinner("Loading models..."):
                app.load_models(TEACHER_PATH, STUDENT_PATH)
    
    # Main area for image processing
    st.header("🎨 Image Enhancement")
    
    if app.models_loaded:
        st.success("✅ Models are loaded and ready!")
    else:
        st.warning("⚠️ Please load models using the button in the sidebar")
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.radio(
            "Select Model:",
            ["student", "teacher"],
            index=0,
            help="Student is faster, Teacher provides higher quality"
        )
        
        if model_type == "teacher":
            st.info("🎯 **Teacher Model**: High-quality enhancement (Slower)")
        else:
            st.info("⚡ **Student Model**: Fast enhancement (Good quality)")
    
    with col2:
        st.markdown("""
        **About the Models:**
        - **Teacher**: Restormer architecture trained for motion deblurring
        - **Student**: GhostNet architecture - lightweight and efficient
        - **Technology**: Knowledge Distillation used to transfer learning
        """)
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Choose an image to enhance",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload a blurry or low-quality image for enhancement"
    )
    
    if uploaded_file is not None:
        # Display original image
        original_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(original_image, use_column_width=True, caption="Input Image")
            
            # Original image info
            st.write(f"**Size**: {original_image.size[0]} × {original_image.size[1]}")
            st.write(f"**Mode**: {original_image.mode}")
        
        with col2:
            st.subheader("✨ Enhanced Image")
            
            if app.models_loaded:
                if st.button("🔮 Enhance Image", type="primary"):
                    with st.spinner("Enhancing image..."):
                        enhanced_image = app.enhance_image(original_image, model_type)
                        
                        if enhanced_image:
                            st.image(enhanced_image, use_column_width=True, caption=f"Enhanced using {model_type.title()} Model")
                            
                            # Enhanced image info
                            st.write(f"**Size**: {enhanced_image.size[0]} × {enhanced_image.size[1]}")
                            
                            # Download button
                            buf = io.BytesIO()
                            enhanced_image.save(buf, format="PNG")
                            st.download_button(
                                label="📥 Download Enhanced Image",
                                data=buf.getvalue(),
                                file_name=f"enhanced_{model_type}.png",
                                mime="image/png"
                            )
            else:
                st.warning("⚠️ Please load models first using the button in the sidebar!")

if __name__ == "__main__":
    main()