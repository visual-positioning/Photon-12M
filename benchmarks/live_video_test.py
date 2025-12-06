import torch
import cv2
import time
import numpy as np
from PIL import Image
import mobileclip
from mobilecap_modern import build_modern_model

# ============================
# CONFIGURATION
# ============================
CHECKPOINT_PATH = "../checkpoints/nano_ep5.pt" 
TOKENIZER_PATH = "../tokenizer/mobilecap_tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.6  # Only show update if we are fairly sure (optional logic)

def main():
    print("--- 🎥 STARTING LIVE NANO DEMO ---")
    
    # 1. Load Models
    print("Loading Models...")
    model, tokenizer = build_modern_model(TOKENIZER_PATH)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    
    clip_model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s1', pretrained='../models/mobileclip_s1.pt')
    clip_model = clip_model.to(DEVICE).eval()

    # 2. Compile for Speed
    print("Compiling (Please wait ~30s)...")
    model = torch.compile(model, mode="reduce-overhead")
    
    # Warmup
    dummy = torch.randn(1, 512).to(DEVICE)
    with torch.no_grad():
        model.generate(dummy, tokenizer, max_new_tokens=10)
    print("Ready!")

    # 3. Webcam Loop
    cap = cv2.VideoCapture(0) # 0 is usually the default camera
    
    # Variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0
    current_caption = "Looking..."
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Optimization: Only caption every 5th frame to keep UI smooth
            # (The model is fast enough to do every frame, but Python UI might lag)
            if frame_count % 5 == 0:
                
                # Convert CV2 (BGR) to PIL (RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                # Inference
                with torch.no_grad():
                    tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
                    
                    # Encode
                    emb = clip_model.encode_image(tensor)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    
                    # Decode
                    current_caption = model.generate(emb, tokenizer, max_new_tokens=15)

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            frame_count += 1

            # --- DRAWING ---
            # Add black bar at bottom
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
            
            # Add Text
            cv2.putText(frame, f"CAPTION: {current_caption}", (10, h-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display
            cv2.imshow('Nano-LlaMA Live', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()