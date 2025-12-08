import torch
import cv2
import time
import queue
import threading
import numpy as np
from PIL import Image
import mobileclip
from mobilecap_modern import build_modern_model

url = # 0 or 1 or http://ip:port/ your stream

CHECKPOINT_PATH = "../checkpoints/nano_ep5.pt"
TOKENIZER_PATH = "../tokenizer/mobilecap_tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAPTION_INTERVAL = 5  # caption every N frames

def inference_thread_fn(frame_q, result_q, model, tokenizer, clip_model, preprocess):
    """Run caption inference asynchronously."""
    while True:
        frame = frame_q.get()
        if frame is None:
            return

        # Convert to RGB (fast OpenCV)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        with torch.no_grad():
            tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE, non_blocking=True)
            emb = clip_model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)

            caption = model.generate(emb, tokenizer, max_new_tokens=15)

        result_q.put(caption)


def main():
    print("\n--- 🚀 STARTING REAL-TIME NANO CAPTIONING ---")

    # Load models
    print("Loading models…")
    model, tokenizer = build_modern_model(TOKENIZER_PATH)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    clip_model, _, preprocess = mobileclip.create_model_and_transforms(
        'mobileclip_s1', pretrained='../models/mobileclip_s1.pt'
    )
    clip_model = clip_model.to(DEVICE).eval()

    # Compile model (huge speedup)
    print("Compiling model…")
    model = torch.compile(model, mode="reduce-overhead")

    # Warmup
    dummy = torch.randn(1, 512).to(DEVICE)
    with torch.no_grad():
        model.generate(dummy, tokenizer, max_new_tokens=5)
    print("Ready!\n")

    # Thread queues
    frame_q = queue.Queue(maxsize=1)
    result_q = queue.Queue(maxsize=1)

    # Start inference thread
    threading.Thread(
        target=inference_thread_fn,
        args=(frame_q, result_q, model, tokenizer, clip_model, preprocess),
        daemon=True
    ).start()

    cap = cv2.VideoCapture(url)  

    current_caption = "Looking..."
    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Resize early (reduces CPU load)
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        # Every N frames, send image to inference thread
        if frame_count % CAPTION_INTERVAL == 0:
            if frame_q.empty():   # avoid backlog
                frame_q.put(frame.copy())

        # Retrieve caption result if available
        if not result_q.empty():
            current_caption = result_q.get()

        # FPS calculation
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        # Draw black bar
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, h - 35), (w, h), (0, 0, 0), -1)

        # Text overlays
        cv2.putText(frame, f"{current_caption}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show window
        cv2.imshow("Real-Time Captioning", frame)

        # Exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    frame_q.put(None)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
