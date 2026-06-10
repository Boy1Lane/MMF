import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class FaceExplainer:
    """
    Explainability module for AI decisions using Grad-CAM algorithm.
    Generates a heatmap highlighting regions the AI focused on the most.
    """
    def __init__(self, model):
        """
        Args:
            model (torch.nn.Module)
        """
        self.model = model
        
        try:
            self.target_layers = [self.model.conv_head]
        except AttributeError:
            # Fallback in case a different architecture is used: move back 2 layers from the end
            layers = list(self.model.children())
            self.target_layers = [layers[-2] if len(layers) >= 2 else layers[-1]]
            
        # Initialize Grad-CAM
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def generate_heatmap(self, face_image_pil, input_tensor):
        """
        Generates Heatmap overlaid onto the original image.
        
        Args:
            face_image_pil (PIL.Image): Original face image (RGB format).
            input_tensor (torch.Tensor): Preprocessed face tensor (batch x C x H x W).
                                        
            
        Returns:
            PIL.Image: Image with red/blue Heatmap overlaid.
        """
        if face_image_pil is None or input_tensor is None:
            return None
            
        # 1. Define prediction target. Since we use num_classes=1 (Binary Classification with BCELoss/Sigmoid)
        # The only output node has index 0.
        targets = [ClassifierOutputTarget(0)]
        
        # 2. Generate grayscale heatmap
        # Pixel values here range between [0, 1]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :] # Get the first image in batch
        
        # 3. Prepare original image
        h_tensor, w_tensor = input_tensor.shape[-2], input_tensor.shape[-1]
        face_img_resized = face_image_pil.resize((w_tensor, h_tensor))
        
        # show_cam_on_image library requires background image as a numpy float array (0.0 - 1.0)
        rgb_img = np.array(face_img_resized, dtype=np.float32) / 255.0
        
        # 4. Overlay both images
        # Red heatmap (regions heavily considered by AI for Deepfake detection) will overlay the face.
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Return as PIL Image for easier display on Streamlit web interface
        return Image.fromarray(visualization)

if __name__ == "__main__":
    print("Test initializing FaceExplainer...")
    print("Done!")
