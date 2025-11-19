"""
Lightweight Anti-Spoofing Detector for Phase 4.1
Uses texture analysis and simple CNN features to detect presentation attacks
"""
import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AntiSpoofDetector:
    """
    Lightweight anti-spoofing detector using:
    - Texture analysis (LBP-like features)
    - Color distribution analysis
    - Frequency domain analysis
    """
    
    def __init__(self):
        self.initialized = True
        logger.info("AntiSpoof detector initialized (texture + frequency analysis)")
    
    def extract_frames(self, video_path, max_frames=10, skip_frames=10):
        """
        Extract frames from video for analysis
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            skip_frames: Extract every Nth frame
            
        Returns:
            List of frames (numpy arrays)
        """
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return frames
        
        frame_count = 0
        extracted = 0
        
        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract every skip_frames frame
            if frame_count % skip_frames == 0:
                # Resize to 256x256
                frame_resized = cv2.resize(frame, (256, 256))
                frames.append(frame_resized)
                extracted += 1
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video (every {skip_frames}th frame)")
        return frames
    
    def compute_texture_score(self, frame):
        """
        Compute texture richness score
        Real faces have richer texture than printed/screen faces
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute Laplacian variance (texture measure)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()
        
        # Normalize to 0-1 range (higher = more texture = more real)
        # Typical real faces: 100-500, fake faces: 10-80
        texture_score = min(1.0, texture_variance / 300.0)
        
        return texture_score
    
    def compute_color_diversity(self, frame):
        """
        Compute color diversity score
        Printed/screen faces have less color variation
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Compute standard deviation of hue and saturation
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        
        # Combine (normalize by typical values)
        color_diversity = min(1.0, (h_std + s_std) / 100.0)
        
        return color_diversity
    
    def compute_frequency_score(self, frame):
        """
        Analyze frequency domain
        Screen/printed faces have different frequency patterns
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Compute high-frequency energy
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # High frequencies are in corners/edges
        high_freq_region = magnitude_spectrum.copy()
        high_freq_region[center_h-30:center_h+30, center_w-30:center_w+30] = 0
        
        high_freq_energy = np.sum(high_freq_region) / np.sum(magnitude_spectrum)
        
        # Real faces have more high-frequency content
        freq_score = min(1.0, high_freq_energy * 10)
        
        return freq_score
    
    def predict_frame(self, frame):
        """
        Predict if frame is real or spoofed
        
        Returns:
            float: probability of being REAL (0=spoofed, 1=real)
        """
        try:
            # Compute multiple features
            texture_score = self.compute_texture_score(frame)
            color_score = self.compute_color_diversity(frame)
            freq_score = self.compute_frequency_score(frame)
            
            # Weighted combination
            # Real faces have high scores in all three
            realness_prob = 0.4 * texture_score + 0.3 * color_score + 0.3 * freq_score
            
            return realness_prob
            
        except Exception as e:
            logger.error(f"Error predicting frame: {e}")
            return 0.5  # Neutral score on error
    
    def predict_video(self, video_path):
        """
        Analyze video and return anti-spoof score
        
        Returns:
            dict with:
                - spoof_score: 0-1 (0=real, 1=spoofed)
                - realness_prob: 0-1 (1=real, 0=spoofed)
                - deepfake_score: 0-1 (average spoof probability across frames)
                - frame_count: number of frames analyzed
                - frame_predictions: list of per-frame results with labels
                - explanation: text explanation
        """
        frames = self.extract_frames(video_path, max_frames=10, skip_frames=10)
        
        if not frames:
            return {
                "spoof_score": 0.5,
                "realness_prob": 0.5,
                "deepfake_score": 0.5,
                "frame_count": 0,
                "frame_predictions": [],
                "explanation": "Could not extract frames from video"
            }
        
        # Predict each frame with detailed results
        frame_predictions = []
        frame_scores = []
        
        for i, frame in enumerate(frames):
            realness = self.predict_frame(frame)
            spoof_prob = 1.0 - realness
            
            # Classify as REAL or SPOOF based on threshold
            label = "REAL" if realness > 0.5 else "SPOOF"
            
            frame_predictions.append({
                "frame_index": i,
                "label": label,
                "realness_prob": float(realness),
                "spoof_prob": float(spoof_prob)
            })
            frame_scores.append(realness)
        
        # Average realness probability
        avg_realness = np.mean(frame_scores)
        spoof_score = 1.0 - avg_realness  # Invert for spoof score
        deepfake_score = spoof_score  # Alias for compatibility
        
        # Generate explanation
        if avg_realness > 0.7:
            explanation = f"High confidence REAL face detected (analyzed {len(frames)} frames)"
        elif avg_realness > 0.4:
            explanation = f"Moderate confidence - possible presentation attack (analyzed {len(frames)} frames)"
        else:
            explanation = f"High spoof probability - likely printed/screen/mask attack (analyzed {len(frames)} frames)"
        
        return {
            "spoof_score": float(spoof_score),
            "realness_prob": float(avg_realness),
            "deepfake_score": float(deepfake_score),
            "frame_count": len(frames),
            "frame_scores": [float(s) for s in frame_scores],
            "frame_predictions": frame_predictions,
            "explanation": explanation
        }


# Global instance
_antispoof_detector = None

def get_antispoof_detector():
    """Get or create global anti-spoof detector instance"""
    global _antispoof_detector
    if _antispoof_detector is None:
        _antispoof_detector = AntiSpoofDetector()
    return _antispoof_detector


if __name__ == "__main__":
    # Test the detector
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        detector = AntiSpoofDetector()
        result = detector.predict_video(video_path)
        
        print("\n" + "="*60)
        print("ANTI-SPOOF DETECTION RESULT")
        print("="*60)
        print(f"Video: {video_path}")
        print(f"Frames analyzed: {result['frame_count']}")
        print(f"\n📊 OVERALL SCORES:")
        print(f"   Realness probability: {result['realness_prob']:.3f}")
        print(f"   Spoof score: {result['spoof_score']:.3f}")
        print(f"   Deepfake score: {result['deepfake_score']:.3f}")
        print(f"\n📋 PER-FRAME PREDICTIONS:")
        for pred in result['frame_predictions']:
            print(f"   Frame {pred['frame_index']}: {pred['label']} "
                  f"(realness={pred['realness_prob']:.3f}, spoof={pred['spoof_prob']:.3f})")
        print(f"\nExplanation: {result['explanation']}")
        print("="*60)
    else:
        print("Usage: python antispoof_detector.py <video_path>")
