import base64
import io
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from PIL import Image

from src.backend.face_detector import FaceDetector
from src.backend.model_loader import DeepfakeModel
from src.backend.explainer import FaceExplainer

app = FastAPI(title="Deepfake Detection API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
detector = None
deepfake_model = None
explainer = None
OPTIMAL_THRESHOLD = 0.78

@app.on_event("startup")
async def startup_event():
    global detector, deepfake_model, explainer
    print("Loading AI Models...")
    detector = FaceDetector()
    
    # Path relative to the project root when running `uvicorn src.backend.main:app`
    model_path = os.path.join("models", "efficientnet_b2_sbi_final.pth")
    if not os.path.exists(model_path):
        print(f"ERROR: Model weight not found at {model_path}")
        return
        
    deepfake_model = DeepfakeModel(model_path=model_path)
    explainer = FaceExplainer(model=deepfake_model.model)
    print("AI Models loaded successfully!")

def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    # Ensure RGB for JPEG format
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

@app.post("/api/analyze")
async def analyze_images(files: List[UploadFile] = File(...)):
    if deepfake_model is None:
        return {"error": "AI Models are not loaded or missing weights."}
        
    results = []
    
    for file in files:
        contents = await file.read()
        try:
            original_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": "Invalid image format."
            })
            continue

        # Step 1: Extract Face
        face_img = detector.extract_face(original_image)
        
        if face_img is None:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": "No face detected in the image. Please try a more frontal angle."
            })
            continue
            
        # Step 2: Predict Forgery Score
        fake_score = deepfake_model.predict(face_img)
        
        # Step 3: Explainability Heatmap
        input_tensor = deepfake_model.transform(face_img).unsqueeze(0).to(deepfake_model.device)
        heatmap_img = explainer.generate_heatmap(face_img, input_tensor)
        
        # Step 4: Format output
        results.append({
            "filename": file.filename,
            "status": "success",
            "original_base64": image_to_base64(original_image),
            "cropped_base64": image_to_base64(face_img),
            "heatmap_base64": image_to_base64(heatmap_img) if heatmap_img else None,
            "fake_score": float(fake_score),
            "is_fake": bool(fake_score >= OPTIMAL_THRESHOLD)
        })
        
    return {"results": results}
