import torch
import torchvision.transforms as T
import timm
from PIL import Image

class DeepfakeModel:
    """
    Module for loading AI models and performing predictions based on trained weights (EfficientNet-B2).
    """
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model on {self.device}...")
        
        # 1. Re-initialize network architecture similar to training phase
        # Note: pretrained=False because we will load our own custom weights
        self.model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=1)
        
        # 2. Load weights from .pth file
        try:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            
        self.model.to(self.device)
        self.model.eval() # Switch to evaluation mode
        
        # 3. Define transformations
        self.transform = T.Compose([
            T.Resize((288, 288)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, face_image):
        """
        Passes face image through the model to predict the Fake Score.
        
        Args:
            face_image (PIL.Image): Face image (usually cropped by face_detector).
            
        Returns:
            float: Fake Score (0.0 -> 1.0).
        """
        if face_image is None:
            return None
            
        # Apply transforms to convert PIL Image to Tensor
        input_tensor = self.transform(face_image).unsqueeze(0) # Add batch dimension (1, C, H, W)
        input_tensor = input_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            with torch.amp.autocast("cuda" if "cuda" in self.device else "cpu"):
                # Model returns raw logits, apply sigmoid to constrain to 0 - 1 range
                logits = self.model(input_tensor).squeeze(1)
                prob = torch.sigmoid(logits).item()
                
        return prob
