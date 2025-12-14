import torch
import cv2
import time
import queue
import threading
import numpy as np
from PIL import Image
import mobileclip
from benchmarks.mobilecap_modern import build_modern_model

url = 0

CHECKPOINT_PATH = "checkpoints/nano_ep5.pt"
TOKENIZER_PATH = "tokenizer/mobilecap_tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAPTION_INTERVAL = 25


def inference_thread_fn(frame_q, result_q, model, tokenizer, clip_model, preprocess):
    """Run caption inference asynchronously and measure detailed timings."""
    while True:
        frame = frame_q.get()
        if frame is None:
            return

        # Convert BGR→RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        with torch.no_grad():

            # Preprocess
            tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE, non_blocking=True)

            # -------------------------------------
            # Measure MobileCLIP inference time
            # -------------------------------------
            t0 = time.time()
            emb = clip_model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            t1 = time.time()
            clip_time = (t1 - t0)

            # -------------------------------------
            # Measure Caption model time
            # -------------------------------------
            t2 = time.time()
            caption = model.generate(emb, tokenizer, max_new_tokens=32)
            t3 = time.time()
            caption_time = (t3 - t2)

        total_time = clip_time + caption_time

        # Send all results back to main thread
        result_q.put((caption, clip_time, caption_time, total_time))


def main():
    print("\n--- 🚀 STARTING REAL-TIME NANO CAPTIONING ---")

    # Load models
    model, tokenizer = build_modern_model(TOKENIZER_PATH)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    clip_model, _, preprocess = mobileclip.create_model_and_transforms(
        'mobileclip_s1', pretrained='../models/mobileclip_s1.pt'
    )
    clip_model = clip_model.to(DEVICE).eval()

    # Compile caption model
    print("Compiling caption model…")
    model = torch.compile(model, mode="reduce-overhead")

    # Warmup
    dummy = torch.randn(1, 512).to(DEVICE)
    model.generate(dummy, tokenizer, max_new_tokens=5)
    print("Ready!\n")

    frame_q = queue.Queue(maxsize=1)
    result_q = queue.Queue(maxsize=1)

    threading.Thread(
        target=inference_thread_fn,
        args=(frame_q, result_q, model, tokenizer, clip_model, preprocess),
        daemon=True
    ).start()

    cap = cv2.VideoCapture(url)

    current_caption = "Looking..."
    clip_time = caption_time = total_time = 0.0

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % CAPTION_INTERVAL == 0:
            if frame_q.empty():
                frame_q.put(frame.copy())

        if not result_q.empty():
            current_caption, clip_time, caption_time, total_time = result_q.get()

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        # Draw info
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)

        cv2.putText(frame, f"{current_caption}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"CLIP: {clip_time*1000:.1f} ms", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.putText(frame, f"Caption: {caption_time*1000:.1f} ms", (180, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.putText(frame, f"Total: {total_time*1000:.1f} ms", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Real-Time Captioning", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    frame_q.put(None)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
