import cv2
import time
import queue
import threading
import torch
from flask import Flask, Response, render_template_string
from PIL import Image
import mobileclip
from mobilecap_modern import build_modern_model

# ----------------------------
# CONFIG
# ----------------------------
STREAM_URL = "rtsp://admin:admin@123@10.23.8.100:554/stream"   # <--- CHANGE YOUR URL HERE
CHECKPOINT_PATH = "../checkpoints/nano_ep5.pt"
TOKENIZER_PATH = "../tokenizer/mobilecap_tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAPTION_INTERVAL = 5

# Flask app
app = Flask(__name__)

# Queues
frame_q = queue.Queue(maxsize=1)
caption_q = queue.Queue(maxsize=1)

stop_flag = False


# ----------------------------
# INFERENCE THREAD (Model)
# ----------------------------
def inference_thread(model, tokenizer, clip_model, preprocess):
    global stop_flag
    while not stop_flag:
        frame = frame_q.get()
        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        with torch.no_grad():
            t = preprocess(pil).unsqueeze(0).to(DEVICE)
            emb = clip_model.encode_image(t)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            caption = model.generate(emb, tokenizer, max_new_tokens=15)

        if caption_q.empty():
            caption_q.put(caption)


# ----------------------------
# FRAME READER THREAD (OpenCV)
# ----------------------------
latest_frame = None

def frame_reader():
    global latest_frame, stop_flag
    cap = cv2.VideoCapture(STREAM_URL)
    frame_count = 0

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count += 1
        latest_frame = frame.copy()

        if frame_count % CAPTION_INTERVAL == 0:
            if frame_q.empty():
                frame_q.put(frame.copy())

    cap.release()


# ----------------------------
# STREAM VIDEO AS MJPEG
# ----------------------------
@app.route("/video_feed")
def video_feed():

    def generate():
        global latest_frame
        while True:
            if latest_frame is None:
                continue

            ret, jpeg = cv2.imencode(".jpg", latest_frame)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg.tobytes() +
                b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ----------------------------
# HOME PAGE (HTML)
# ----------------------------
@app.route("/")
def home():
    html = """
    <html>
    <head>
        <title>Nano Captioning Web</title>
        <style>
            body { font-family: Arial; text-align: center; padding: 20px; }
            #caption-box {
                margin-top: 20px;
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }
        </style>
    </head>
    <body>
        <h2>Nano Real-Time Captioning</h2>

        <img src="/video_feed" width="640">

        <div id="caption-box">Waiting...</div>

        <script>
            function pollCaption() {
                fetch("/caption")
                .then(response => response.text())
                .then(text => { 
                    if (text.trim() !== "") {
                        document.getElementById("caption-box").innerText = text;
                    }
                });
            }
            setInterval(pollCaption, 200);
        </script>

    </body>
    </html>
    """
    return render_template_string(html)


# ----------------------------
# GET CAPTION
# ----------------------------
@app.route("/caption")
def caption():
    if not caption_q.empty():
        return caption_q.get()
    return ""


# ----------------------------
# SERVER MAIN
# ----------------------------
if __name__ == "__main__":
    print("Loading models...")
    model, tokenizer = build_modern_model(TOKENIZER_PATH)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    model = torch.compile(model)

    clip_model, _, preprocess = mobileclip.create_model_and_transforms(
        "mobileclip_s1", pretrained="../models/mobileclip_s1.pt"
    )
    clip_model = clip_model.to(DEVICE).eval()

    threading.Thread(target=inference_thread, args=(model, tokenizer, clip_model, preprocess), daemon=True).start()
    threading.Thread(target=frame_reader, daemon=True).start()

    print("Server running at http://127.0.0.1:5093")
    app.run(host="0.0.0.0", port=5093, threaded=True)
