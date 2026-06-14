# Deepfake Detection using Self-Blended Images (SBI)

## Objective
The main objective of this project is to develop a robust Deepfake Detection system that can accurately identify forged or manipulated facial images. The system utilizes an EfficientNet-B2 model trained on a self-generated dataset using the Self-Blended Images (SBI) technique. Furthermore, it incorporates Explainable AI (XAI) through Grad-CAM to provide visual evidence (heatmaps) highlighting the specific facial regions that influenced the model's decision, making the detection process transparent and interpretable.

## Architecture
The system architecture consists of several core modules working sequentially to process an image and output a detection result:

1. **Face Alignment & Cropping**: 
   - Detects the facial region in the uploaded image.
   - Aligns and crops the face to a standardized size, ignoring irrelevant background information to help the model focus purely on facial features.

2. **Self-Blended Image (SBI) Module (Training Phase)**:
   - Used primarily during the model training phase.
   - Generates synthetic "fake" images by blending modified facial masks back onto real images. This teaches the model to recognize blending boundaries and frequency inconsistencies typical of deepfakes without relying on pre-existing deepfake datasets.

3. **Detection Model**:
   - The core AI engine, powered by a pre-trained **EfficientNet-B2** architecture.
   - It takes the cropped facial image as input and outputs a "Fake Score" indicating the probability of the image being manipulated.

4. **Explainability (XAI) with Grad-CAM**:
   - Applies Gradient-weighted Class Activation Mapping (Grad-CAM) to the final convolutional layer of the EfficientNet model.
   - Generates a visual heatmap overlaid on the original face, where red areas indicate abnormal traces (manipulations) that the model focused on to make its prediction.

5. **Web Interface**:
   - Built with **Streamlit** to provide a user-friendly, interactive UI.
   - Handles image uploads, orchestrates the AI pipeline, and beautifully renders the original image, cropped face, explainability heatmap, and the final evaluation score.

## Environment Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Boy1Lane/MMF.git
   ```
2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```
3. **Install Dependencies**:
   Install the required libraries, including PyTorch with GPU support (CUDA 12.1).
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### 1. Prepare the Model Weights
Ensure the pre-trained model weights are correctly placed before running the application.
1. Create a `models` directory in the root of the project.
2. Place the trained model weights file into the `models/` folder.
3. Ensure the filename matches what the app expects: `efficientnet_b2_sbi_final.pth`.

### 2. Start the Application
Run the Streamlit web application using the following command:
```bash
python -m streamlit run app.py
```

### 3. Usage
- The command above will start a local server and automatically open a tab in your default web browser (usually at `http://localhost:8501`).
- Click "Browse files" to upload an image containing a face (JPG, JPEG, PNG).
- The system will process the image and display the original image, the extracted face, the Grad-CAM heatmap, and the final authenticity evaluation result.
