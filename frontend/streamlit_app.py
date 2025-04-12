import streamlit as st
import requests
from PIL import Image
import io
import base64
import time
import os

# Configure environment
API_URL = os.getenv("API_URL", "http://localhost:8000")

# App title and configuration
st.set_page_config(
    page_title="Refaceit - AI Facial Enhancement",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 24px;
        margin-bottom: 20px;
    }
    .success-msg {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        margin-bottom: 20px;
    }
    .info-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #e6f7ff;
        border: 1px solid #b3d7ff;
        margin-bottom: 15px;
    }
    .section-header {
        font-size: 20px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main app header
st.markdown('<p class="main-header">AI Facial Enhancement</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Using GFPGAN - State-of-the-Art GAN-based Face Restoration</p>', unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses GFPGAN, a state-of-the-art GAN-based model for facial restoration and enhancement developed by Tencent ARC Lab.
    
    GFPGAN can:
    - Restore facial details and enhance overall quality
    - Fix blurry or low-resolution faces 
    - Improve lighting and coloration
    - Generate natural-looking results
    """)
    
    st.header("How it works")
    st.markdown("""
    1. **Upload**: Submit your image containing one or more faces
    2. **Analysis**: GFPGAN detects faces in the image
    3. **Enhancement**: The model applies restoration techniques to each face
    4. **Result**: View and download the enhanced image
    """)
    
    st.header("Technical Details")
    st.markdown("""
    - **Model**: GFPGAN v1.3
    - **Architecture**: Generative Adversarial Network (GAN)
    - **Backend**: FastAPI with PyTorch
    - **Frontend**: Streamlit
    - **Infrastructure**: Containerized with Docker
    - **Monitoring**: Prometheus & Grafana
    """)
    
    # Add a health check indicator
    st.header("System Status")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            st.success("✅ Backend API: Online")
        else:
            st.error("❌ Backend API: Error")
    except:
        st.error("❌ Backend API: Unreachable")

# Main content
col1, col2 = st.columns(2)

# Upload section
with col1:
    st.header("Original Image")
    uploaded_file = st.file_uploader("Upload an image with faces", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        # Image info
        st.markdown(f"**Image size**: {image.size[0]} x {image.size[1]} pixels")
        
        # Technical tips
        with st.expander("📌 Tips for best results"):
            st.markdown("""
            - For optimal results, use images where faces are clearly visible
            - The model works best on front-facing portraits with good lighting
            - If faces are very small or obscured, results may vary
            - Processing time depends on image size and number of faces
            """)

# Results section
with col2:
    st.header("Enhanced Result")
    
    if uploaded_file is not None:
        # Process button
        if st.button("Enhance Image"):
            with st.spinner("Processing image with GFPGAN..."):
                try:
                    # Reset file pointer and prepare for upload
                    uploaded_file.seek(0)
                    
                    # Record start time
                    start_time = time.time()
                    
                    # Send request to API
                    response = requests.post(
                        f"{API_URL}/enhance/", 
                        files={"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
                    )
                    
                    # Calculate client-side processing time
                    client_time = time.time() - start_time
                    
                    # Check response
                    if response.status_code == 200:
                        # Get the enhanced image
                        result = response.json()
                        enhanced_image_b64 = result["enhanced_image"]
                        server_time = result.get("processing_time", 0)
                        
                        # Convert base64 to image
                        enhanced_image = Image.open(io.BytesIO(base64.b64decode(enhanced_image_b64)))
                        
                        # Display enhanced image
                        st.image(enhanced_image, use_column_width=True)
                        
                        # Display processing time
                        st.markdown(f"""
                        <div class="success-msg">
                            ✨ Processing complete!<br>
                            ⏱️ Server time: {server_time:.2f}s | Total time: {client_time:.2f}s
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add download button
                        buf = io.BytesIO()
                        enhanced_image.save(buf, format="PNG")
                        st.download_button(
                            label="Download Enhanced Image",
                            data=buf.getvalue(),
                            file_name="enhanced_image.png",
                            mime="image/png"
                        )
                        
                        # Show what was improved
                        with st.expander("🔍 What was improved?"):
                            st.markdown("""
                            The GFPGAN model enhances faces by:
                            - Improving facial details and textures
                            - Enhancing facial features like eyes, nose, and lips
                            - Smoothing skin while maintaining natural texture
                            - Adjusting lighting and contrast
                            - Preserving the overall identity of the person
                            """)
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to the backend service: {e}")
    else:
        st.info("👈 Please upload an image to see the enhancement")

# MLOps explanation
st.markdown("---")
st.markdown('<p class="section-header">MLOps Implementation Details</p>', unsafe_allow_html=True)

# Three-column layout for MLOps details
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Model")
    st.markdown("""
    - **GFPGAN v1.3** by Tencent ARC Lab
    - Pre-trained GAN-based restoration model
    - Specialized for facial enhancement
    - Face detection and region-based processing
    """)

with col2:
    st.markdown("### API & Infrastructure")
    st.markdown("""
    - FastAPI backend for model serving
    - RESTful API with proper error handling
    - Docker containerization for portability
    - Prometheus metrics for monitoring
    - Optimized inference pipeline
    """)

with col3:
    st.markdown("### Deployment")
    st.markdown("""
    - Containerized with Docker Compose
    - Ready for cloud deployment (Azure/GCP)
    - CI/CD compatible architecture
    - Scalable design for production use
    - Monitoring and health checks
    """)

# Footer with attribution
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px; padding: 10px;">
<p>© 2025 Refaceit- AI Facial Enhancement Demo | GFPGAN developed by <a href="https://github.com/TencentARC/GFPGAN" target="_blank">Tencent ARC Lab</a></p>
</div>
""", unsafe_allow_html=True)
