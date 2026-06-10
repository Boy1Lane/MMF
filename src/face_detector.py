import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

class FaceDetector:
    """
    Face detection and cropping module using MTCNN.
    """
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize MTCNN
        self.mtcnn = MTCNN(
            image_size=256,
            margin=32,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=self.device
        )

    def extract_face(self, image):
        """
        Detects and crops a face from an image.
        
        Args:
            image (PIL.Image or numpy.ndarray): Input image (RGB).
            
        Returns:
            PIL.Image: Cropped face image (256x256) or None if no face is found.
        """
        # Ensure input image is in PIL format
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image.convert("RGB")

        # MTCNN will automatically detect and crop the face
        face_tensor = self.mtcnn(pil_img)

        if face_tensor is not None:
            # Convert returned tensor (C, H, W) to numpy array (H, W, C)
            face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
            
            # Clip values to standard 0-255 range for RGB images
            face_np = np.clip(face_np, 0, 255).astype(np.uint8)
            
            # Return as PIL Image for easy display in Streamlit later
            return Image.fromarray(face_np)
            
        return None
