import torch
import time
import os
import logging
from PIL import Image
import mobileclip
from mobilecap_modern import build_modern_model

# ============================
# 1. CONFIGURATION
# ============================
CHECKPOINT_PATH = "../checkpoints/nano_ep5.pt" 
TOKENIZER_PATH = "../tokenizer/mobilecap_tokenizer.json"
REAL_IMAGE_PATH = r"C:\Dataset\coco2017\val2017\000000000139.jpg" # Update this to ANY real jpg you have
LOG_FILE = "speed_log.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# 2. LOGGER SETUP
# ============================
# This sets up a logger that writes to BOTH console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'), # 'w' overwrites each time
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def main():
    logger.info("==========================================")
    logger.info("       MOBILECAP NANO SPEED BENCHMARK     ")
    logger.info("==========================================")
    logger.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # 1. Load Model
    logger.info("\n[1] Loading Model...")
    model, tokenizer = build_modern_model(TOKENIZER_PATH)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    
    # Vision Encoder
    clip_model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s1', pretrained='../models/mobileclip_s1.pt')
    clip_model = clip_model.to(DEVICE).eval()

    # 2. COMPILE (The Turbo Button)
    logger.info("[2] Compiling Model (Torch 2.0 Optimization)...")
    logger.info("    (This takes ~1 minute but doubles speed)")
    t_start_compile = time.time()
    # We only compile the decoder because CLIP is already fast enough
    model = torch.compile(model, mode="reduce-overhead")
    
    # Warmup compilation with dummy data
    dummy_emb = torch.randn(1, 512).to(DEVICE)
    with torch.no_grad():
        model.generate(dummy_emb, tokenizer, max_new_tokens=5)
    logger.info(f"    -> Compilation Finished in {time.time() - t_start_compile:.2f}s")

    # ==========================================
    # TEST A: THE "FERRARI" TEST (Pure Throughput)
    # Uses Dummy Data to measure raw GPU limits
    # ==========================================
    BATCH_SIZE = 512
    logger.info(f"\n[3] RUNNING RAW THROUGHPUT TEST (Batch Size: {BATCH_SIZE})")
    logger.info("    (Measures how many images per second the GPU can handle in parallel)")
    
    dummy_batch = torch.randn(BATCH_SIZE, 512).to(DEVICE)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad(): model.generate(dummy_batch, tokenizer, max_new_tokens=20)
        
    torch.cuda.synchronize()
    t0 = time.time()
    
    with torch.no_grad():
        # Generate 20 tokens (standard caption length)
        model.generate(dummy_batch, tokenizer, max_new_tokens=20)
        
    torch.cuda.synchronize()
    t1 = time.time()
    
    total_time = t1 - t0
    fps = BATCH_SIZE / total_time
    
    logger.info(f"    -> Time for {BATCH_SIZE} captions: {total_time*1000:.2f} ms")
    logger.info(f"    -> Raw Throughput: {fps:.2f} FPS")
    logger.info(f"    -> Latency per Image (Amortized): {(total_time/BATCH_SIZE)*1000:.2f} ms")

    # ==========================================
    # TEST B: THE "REAL WORLD" TEST (End-to-End)
    # Includes Disk Read -> Preprocess -> Encode -> Decode
    # ==========================================
    logger.info(f"\n[4] RUNNING REAL-WORLD LATENCY TEST (Batch Size: 1)")
    logger.info("    (Measures time from 'Opening Image' to 'Text Output')")
    
    if os.path.exists(REAL_IMAGE_PATH):
        # Time the whole pipeline
        torch.cuda.synchronize()
        t0 = time.time()
        
        # Step A: Load from Disk
        image = Image.open(REAL_IMAGE_PATH).convert("RGB")
        
        # Step B: Preprocess
        img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        
        # Step C: Vision Encode
        with torch.no_grad():
            img_emb = clip_model.encode_image(img_tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            
            # Step D: Text Decode
            caption = model.generate(img_emb, tokenizer, max_new_tokens=20)
            
        torch.cuda.synchronize()
        t1 = time.time()
        
        logger.info(f"    -> Total End-to-End Time: {(t1-t0)*1000:.2f} ms")
        logger.info(f"    -> Generated Caption: \"{caption}\"")
    else:
        logger.warning(f"    [!] Could not find real image at {REAL_IMAGE_PATH}. Skipping Test B.")

    logger.info("\nDone. Results saved to speed_log.txt")

if __name__ == "__main__":
    main()