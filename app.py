import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Deepfake Detection SBI", layout="wide")

# ========================================================
# CACHED MODEL LOADING FUNCTION (SPEEDS UP APP)
# ========================================================
@st.cache_resource
def load_models():
    # Delay heavy imports so the UI renders immediately
    import torch
    from src.face_detector import FaceDetector
    from src.model_loader import DeepfakeModel
    from src.explainer import FaceExplainer

    # 1. Initialize face detector
    detector = FaceDetector()
    
    # 2. Initialize prediction model
    model_path = os.path.join("models", "efficientnet_b2_sbi_final.pth")
    
    # Quick check if file exists
    if not os.path.exists(model_path):
        return None, None, None, f"Weight file not found at: {model_path}. Please copy the file from Colab to this directory!"
        
    deepfake_model = DeepfakeModel(model_path=model_path)
    
    # 3. Initialize Heatmap explainer
    explainer = FaceExplainer(model=deepfake_model.model)
    
    return detector, deepfake_model, explainer, "Success"

# ========================================================
# MAIN UI
# ========================================================

# Inject custom CSS to make success/error/warning bars fit their content
st.markdown("""
    <style>
    div[data-testid="stAlert"] {
        width: fit-content;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="background: -webkit-linear-gradient(45deg, #FF4B4B, #FF8C00); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0;">
        Deepfake Detection Engine
    </h1>
    <p style="font-size: 1.2rem; color: #888; font-weight: 500;">Powered by Self-Blended Images (SBI) & EfficientNet</p>
</div>
""", unsafe_allow_html=True)

# Wait for models to load before showing UI
with st.spinner("Loading AI Models (takes a few seconds initially)..."):
    detector, deepfake_model, explainer, status = load_models()
    
if status != "Success":
    st.error(status)
    st.stop() # Stop execution if model is missing

st.markdown("---")

st.markdown("### Upload Images for Analysis")
st.caption("The AI will automatically extract the face, evaluate the forgery score, and generate an explainability heatmap.")
uploaded_files = st.file_uploader("Select images to check...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="collapsed")

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown(f"<h3 style='color: #4CAF50; margin-top: 2rem;'>🔎 Analysis Report: <code>{uploaded_file.name}</code></h3>", unsafe_allow_html=True)
        
        # Read original image using PIL
        original_image = Image.open(uploaded_file).convert("RGB")
        
        # Add 2 spacer columns on the sides to squeeze the 3 main columns into the center
        spacer_left, col1, col2, col3, spacer_right = st.columns([0.5, 2.5, 2.5, 2.5, 0.5])
        
        with col1:
            st.markdown("**1. Original**")
            st.image(original_image, width="stretch")
            
        with st.spinner(f"AI is analyzing {uploaded_file.name}..."):
            # Step 1: Extract face
            face_img = detector.extract_face(original_image)
            
            if face_img is None:
                with col1:
                    st.warning("No face detected in the image. Please try a more frontal angle.")
            else:
                with col2:
                    st.markdown("**2. Cropped Face**")
                    st.image(face_img, width="stretch")
                    
                # Step 2: Predict
                fake_score = deepfake_model.predict(face_img)
                
                # Step 3: Heatmap (Convert image to Tensor exactly like training phase)
                input_tensor = deepfake_model.transform(face_img).unsqueeze(0).to(deepfake_model.device)
                heatmap_img = explainer.generate_heatmap(face_img, input_tensor)
                
                with col3:
                    st.markdown("**3. Explainability (Grad-CAM)**")
                    if heatmap_img:
                        st.image(heatmap_img, width="stretch")
                        st.caption(":red[Red]: Abnormal traces | :blue[Blue]: Normal")

                st.markdown("---")
                
                # Split evaluation into 2 columns for a better layout
                res_col1, res_col2 = st.columns([3, 1])
                
                OPTIMAL_THRESHOLD = 0.78
                
                with res_col1:
                    st.subheader("Evaluation Results")
                    if fake_score >= OPTIMAL_THRESHOLD:
                        st.error(f"**DEEPFAKE DETECTED**")
                        st.markdown(f"The AI found strong evidence of manipulation. *(Threshold: {OPTIMAL_THRESHOLD})*")
                    else:
                        st.success(f"**AUTHENTIC**")
                        st.markdown(f"The AI did not find significant manipulation traces. *(Threshold: {OPTIMAL_THRESHOLD})*")
                        
                    st.progress(fake_score)
                    
                with res_col2:
                    # Beautiful metric display
                    st.metric(
                        label="Forgery Confidence", 
                        value=f"{fake_score * 100:.1f}%", 
                        delta="Suspicious" if fake_score >= OPTIMAL_THRESHOLD else "Natural",
                        delta_color="inverse"
                    )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
