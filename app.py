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

st.title("Deepfake Detection (Self-Blended Images)")
st.write("Upload an image containing a face. The AI will automatically crop it, evaluate the forgery score, and generate an explainability heatmap.")

# Wait for models to load before showing UI
with st.spinner("Loading AI Models (takes a few seconds initially)..."):
    detector, deepfake_model, explainer, status = load_models()
    
if status != "Success":
    st.error(status)
    st.stop() # Stop execution if model is missing

st.markdown("---")

uploaded_files = st.file_uploader("Select images to check...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown(f"### Results for: `{uploaded_file.name}`")
        
        # Read original image using PIL
        original_image = Image.open(uploaded_file).convert("RGB")
        
        # Add 2 spacer columns on the sides to squeeze the 3 main columns into the center
        spacer_left, col1, col2, col3, spacer_right = st.columns([1, 2, 2, 2, 1])
        
        with col1:
            st.subheader("1. Original")
            st.image(original_image, width="stretch")
            
        with st.spinner(f"AI is analyzing {uploaded_file.name}..."):
            # Step 1: Extract face
            face_img = detector.extract_face(original_image)
            
            if face_img is None:
                st.warning("No face detected in the image. Please try a more frontal angle.")
            else:
                with col2:
                    st.subheader("2. Cropped Face")
                    st.image(face_img, width="stretch")
                    
                # Step 2: Predict
                fake_score = deepfake_model.predict(face_img)
                
                # Step 3: Heatmap (Convert image to Tensor exactly like training phase)
                input_tensor = deepfake_model.transform(face_img).unsqueeze(0).to(deepfake_model.device)
                heatmap_img = explainer.generate_heatmap(face_img, input_tensor)
                
                with col3:
                    st.subheader("3. Explainability (Grad-CAM)")
                    if heatmap_img:
                        st.image(heatmap_img, width="stretch")
                        st.caption("Red regions: Abnormal traces (manipulation). Blue regions: Normal.")

                st.markdown("---")
                st.subheader("Evaluation Results")
                
                OPTIMAL_THRESHOLD = 0.78
                
                if fake_score >= OPTIMAL_THRESHOLD:
                    st.error(f"DEEPFAKE DETECTED (Fake Score: {fake_score:.4f})")
                else:
                    st.success(f"AUTHENTIC. (Fake Score: {fake_score:.4f})")
                    
                st.progress(fake_score)
                st.write(f"*Current evaluation threshold is set to: {OPTIMAL_THRESHOLD}*")
        
        st.markdown("---")
