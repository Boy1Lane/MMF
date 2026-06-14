# Deepfake Detection using Self-Blended Images (SBI)

## Objective
The main objective of this project is to develop a robust Deepfake Detection system that can accurately identify forged or manipulated facial images. The system utilizes an EfficientNet-B2 model trained on a self-generated dataset using the Self-Blended Images (SBI) technique. Furthermore, it incorporates Explainable AI (XAI) through Grad-CAM to provide visual evidence (heatmaps) highlighting the specific facial regions that influenced the model's decision, making the detection process transparent and interpretable.

## Architecture
The system architecture consists of a modern, decoupled web stack to process images and output detection results:

1. **Face Alignment & Cropping**: 
   - Detects the facial region in the uploaded image.
   - Aligns and crops the face to a standardized size, ignoring irrelevant background information to help the model focus purely on facial features.

2. **Self-Blended Image (SBI) Module (Training Phase)**:
   - Generates synthetic "fake" images by blending modified facial masks back onto real images. This teaches the model to recognize blending boundaries and frequency inconsistencies typical of deepfakes without relying on pre-existing deepfake datasets.

3. **Detection Model**:
   - The core AI engine, powered by a pre-trained **EfficientNet-B2** architecture.
   - It takes the cropped facial image as input and outputs a "Fake Score" indicating the probability of the image being manipulated.

4. **Explainability (XAI) with Grad-CAM**:
   - Applies Gradient-weighted Class Activation Mapping (Grad-CAM) to the final convolutional layer of the EfficientNet model.
   - Generates a visual heatmap overlaid on the original face, where red areas indicate abnormal traces (manipulations).

5. **Decoupled Web Application**:
   - **Backend (FastAPI)**: A high-performance Python backend serving the AI models. It exposes a REST API (`/api/analyze`) that processes uploaded images and returns base64-encoded results.
   - **Frontend (React.js + Vite)**: A premium, dynamic user interface built with React. It handles drag-and-drop file uploads, orchestrates API calls, and beautifully renders the analysis reports with smooth micro-animations.

## Environment Setup

### Prerequisites
- Python 3.9+
- Node.js (v16+) and npm

### 1. Clone the Repository
```bash
git clone https://github.com/Boy1Lane/MMF.git
cd MMF
```

### 2. Backend Setup (Python)
Create a virtual environment and install dependencies:
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements (includes PyTorch, FastAPI, etc.)
pip install -r requirements.txt
```

### 3. Frontend Setup (React/Node)
Navigate to the frontend directory and install NPM packages:
```bash
cd src/frontend
npm install
```

## How to Run

Because the application is decoupled, you must start **two separate servers**.

### 1. Prepare the Model Weights
1. Ensure a `models` directory exists in the root of the project.
2. Place the trained model weights file (`efficientnet_b2_sbi_final.pth`) into the `models/` folder.

### 2. Start the Backend (FastAPI)
Open a terminal in the **root** `MMF` directory, ensure your virtual environment is activated, and run:
```bash
python -m uvicorn src.backend.main:app --reload
```
*(The API will start at `http://localhost:8000`)*

### 3. Start the Frontend (React.js)
Open a **second** terminal, navigate to the frontend folder, and run:
```bash
cd src/frontend
npm run dev
```
*(The frontend will start at `http://localhost:5173`. Open this link in your browser to access the UI.)*

### 4. Usage
- Drag and drop one or more images (JPG, PNG) into the upload zone.
- Click the **Start Processing** button.
- The system will analyze the images and display a staged, animated report showing the Original Image, Cropped Face, Explainability Heatmap, and the final Authenticity Score.
