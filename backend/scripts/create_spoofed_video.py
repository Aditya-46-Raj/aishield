"""
Create a synthetic spoofed video by applying screen replay effects
"""
import cv2
import numpy as np
from pathlib import Path

def create_spoofed_video(input_path, output_path):
    """
    Create spoofed video by:
    1. Adding moiré patterns (screen effect)
    2. Reducing texture detail (printed/screen effect)
    3. Adding blur to simulate replay attack
    4. Reducing color diversity
    """
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply spoofing effects
        
        # 1. Add stronger moiré pattern (screen effect)
        moire = np.zeros_like(frame)
        for i in range(0, height, 3):
            for j in range(0, width, 3):
                moire[i:i+2, j:j+2] = [20, 20, 20]
        frame = cv2.addWeighted(frame, 0.85, moire, 0.15, 0)
        
        # 2. Reduce texture significantly (blur to simulate print/screen)
        frame = cv2.GaussianBlur(frame, (11, 11), 3.0)
        
        # 3. Reduce color diversity significantly (flatten colors)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = (hsv[:, :, 1] * 0.4).astype(np.uint8)  # Reduce saturation more
        hsv[:, :, 0] = (hsv[:, :, 0] * 0.7).astype(np.uint8)  # Reduce hue diversity
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 4. Add stronger brightness reduction (screen effect)
        frame = cv2.addWeighted(frame, 0.75, np.zeros_like(frame), 0, 5)
        
        # 5. Add stronger compression artifacts (JPEG-like)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
        _, encimg = cv2.imencode('.jpg', frame, encode_param)
        frame = cv2.imdecode(encimg, 1)
        
        # 6. Reduce high-frequency content (flattens texture)
        frame = cv2.bilateralFilter(frame, 9, 75, 75)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"✅ Created spoofed video: {output_path}")
    print(f"   Frames processed: {frame_count}")
    print(f"   Effects applied: strong moiré, blur, color reduction, compression, bilateral filter")
    
    return True

if __name__ == "__main__":
    input_video = Path("../samples/video_sample.mp4")
    output_video = Path("../samples/video_spoofed.mp4")
    
    if not input_video.exists():
        print(f"❌ Input video not found: {input_video}")
        exit(1)
    
    create_spoofed_video(input_video, output_video)
