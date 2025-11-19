"""
Deepfake Detection Module
Uses pretrained EfficientNet-B0 for detecting deepfake/synthetic faces in video frames
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DeepfakeDetector:
    """Lightweight deepfake detector using EfficientNet-B0"""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize deepfake detector
        Args:
            model_path: Path to pretrained model weights (optional)
            device: torch device (cuda/cpu)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create EfficientNet-B0 for binary classification (real vs fake)
        self.model = models.efficientnet_b0(pretrained=True)
        
        # Replace classifier for binary classification
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )
        
        # If pretrained weights provided, load them
        if model_path and Path(model_path).exists():
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded deepfake detector from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load pretrained weights: {e}, using ImageNet features")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Deepfake detector initialized on {self.device}")
    
    def predict_frame(self, frame):
        """
        Predict if a single frame is deepfake
        Args:
            frame: numpy array (BGR) or PIL Image
        Returns:
            float: deepfake probability [0,1] where 1 = likely fake
        """
        try:
            # Convert to PIL if needed
            if isinstance(frame, np.ndarray):
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
            
            # Preprocess
            img_tensor = self.transform(frame).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(img_tensor)
                prob = float(output.item())
            
            return prob
            
        except Exception as e:
            logger.error(f"Frame prediction error: {e}")
            return 0.5  # neutral score on error
    
    def analyze_video(self, video_path, sample_rate=1.0):
        """
        Analyze video for deepfake content
        Args:
            video_path: path to video file
            sample_rate: frames per second to sample (default 1 fps)
        Returns:
            dict with deepfake_score, frame_scores, stats
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {
                    "deepfake_score": 0.5,
                    "error": "Could not open video",
                    "frames_analyzed": 0
                }
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30  # fallback
            
            frame_interval = max(1, int(fps / sample_rate))
            
            frame_scores = []
            frame_count = 0
            analyzed_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames based on interval
                if frame_count % frame_interval == 0:
                    score = self.predict_frame(frame)
                    frame_scores.append(score)
                    analyzed_count += 1
                
                frame_count += 1
            
            cap.release()
            
            if not frame_scores:
                return {
                    "deepfake_score": 0.5,
                    "frames_analyzed": 0,
                    "error": "No frames analyzed"
                }
            
            # Calculate statistics
            avg_score = np.mean(frame_scores)
            max_score = np.max(frame_scores)
            std_score = np.std(frame_scores)
            
            return {
                "deepfake_score": float(avg_score),
                "max_deepfake_score": float(max_score),
                "std_deviation": float(std_score),
                "frames_analyzed": analyzed_count,
                "total_frames": frame_count,
                "sample_rate_fps": sample_rate,
                "frame_scores": frame_scores[:10]  # Return first 10 for inspection
            }
            
        except Exception as e:
            logger.exception(f"Video analysis error: {e}")
            return {
                "deepfake_score": 0.5,
                "error": str(e),
                "frames_analyzed": 0
            }


# Singleton instance
_detector_instance = None

def get_deepfake_detector(model_path=None):
    """Get or create singleton deepfake detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = DeepfakeDetector(model_path=model_path)
    return _detector_instance


if __name__ == "__main__":
    # Test the detector
    print("Testing Deepfake Detector...")
    detector = DeepfakeDetector()
    
    # Test with a sample video if available
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        result = detector.analyze_video(video_path, sample_rate=1.0)
        print("\nResults:")
        print(f"  Deepfake Score: {result['deepfake_score']:.3f}")
        print(f"  Frames Analyzed: {result['frames_analyzed']}")
        print(f"  Max Score: {result.get('max_deepfake_score', 0):.3f}")
    else:
        print("Usage: python deepfake_detector.py <video_path>")
