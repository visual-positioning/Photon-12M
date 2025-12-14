import torch
import time
import os
import logging
from PIL import Image
import mobileclip
from mobilecap_modern import build_modern_model

# ============================
# CONFIG
# ============================
CHECKPOINT_PATH = "../checkpoints/nano_ep5.pt"
TOKENIZER_PATH = "../tokenizer/mobilecap_tokenizer.json"
REAL_IMAGE_PATH = r"C:\Dataset\coco2017\val2017\000000000139.jpg"  # set to a real image
LOG_FILE = "speed_log.txt"
USE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================
# LOGGER (console + file)
# =========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# =========================================
# UTILS
# =========================================
def sync_device(device):
    if device == "cuda":
        torch.cuda.synchronize()

def safe_to_str(cap):
    # model.generate may return a Python string, list, or tensor-like object.
    if isinstance(cap, str):
        return cap
    try:
        # try common tokenizer-like decode
        if hasattr(cap, "tolist"):
            tokens = cap.tolist()
            return str(tokens)
    except Exception:
        pass
    try:
        return str(cap)
    except Exception:
        return "<unable to stringify caption>"

# =========================================
# MAIN
# =========================================
def main():
    logger.info("==========================================")
    logger.info("       MOBILECAP NANO SPEED BENCHMARK     ")
    logger.info("==========================================")
    if USE_DEVICE == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "CPU"
    logger.info(f"Device chosen: {USE_DEVICE} ({device_name})")

    # -------------------------
    # Load caption model
    # -------------------------
    logger.info("\n[1] Loading caption model...")
    model, tokenizer = build_modern_model(TOKENIZER_PATH)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=USE_DEVICE))
    model.to(USE_DEVICE).eval()

    # -------------------------
    # Load CLIP model + preprocess
    # -------------------------
    logger.info("[2] Loading MobileCLIP...")
    clip_model, _, preprocess = mobileclip.create_model_and_transforms(
        "mobileclip_s1", pretrained="../models/mobileclip_s1.pt"
    )
    clip_model.to(USE_DEVICE).eval()

    # -------------------------
    # Compile caption model (optional)
    # -------------------------
    logger.info("[3] Compiling caption model (torch.compile)...")
    try:
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("    -> torch.compile succeeded")
    except Exception as e:
        logger.warning(f"    -> torch.compile skipped / failed: {e}")

    # Warmup (important for stable timings)
    logger.info("[4] Warmup (compilation + caches)...")
    dummy_emb = torch.randn(1, 512).to(USE_DEVICE)
    with torch.no_grad():
        # warmup decoder
        try:
            model.generate(dummy_emb, tokenizer, max_new_tokens=5)
        except Exception:
            # If generate throws (some models expect different input shape), ignore warmup
            logger.warning("    -> Warmup generate failed (ignored).")

    # -------------------------
    # TEST A: RAW THROUGHPUT (dummy embeddings)
    # -------------------------
    BATCH_SIZE = 512
    logger.info(f"\n[5] RAW THROUGHPUT TEST (batch={BATCH_SIZE})")
    dummy_batch = torch.randn(BATCH_SIZE, 512).to(USE_DEVICE)

    # Warmup a few times
    try:
        for _ in range(3):
            with torch.no_grad():
                _ = model.generate(dummy_batch, tokenizer, max_new_tokens=20)
        sync_device(USE_DEVICE)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(dummy_batch, tokenizer, max_new_tokens=20)
        sync_device(USE_DEVICE)
        t1 = time.perf_counter()
        total_time = t1 - t0
        fps = BATCH_SIZE / total_time
        logger.info(f"    -> Time for {BATCH_SIZE} captions: {total_time*1000:.2f} ms")
        logger.info(f"    -> Raw Throughput: {fps:.2f} images/sec")
        logger.info(f"    -> Amortized Latency/image: {(total_time/BATCH_SIZE)*1000:.4f} ms")
    except RuntimeError as e:
        logger.warning(f"    -> OOM or runtime error with batch={BATCH_SIZE}: {e}")
        # try a smaller batch
        for fallback in (256, 128, 64, 32):
            try:
                logger.info(f"    -> Trying fallback batch {fallback}")
                dummy_batch = torch.randn(fallback, 512).to(USE_DEVICE)
                for _ in range(2):
                    with torch.no_grad():
                        _ = model.generate(dummy_batch, tokenizer, max_new_tokens=20)
                sync_device(USE_DEVICE)
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model.generate(dummy_batch, tokenizer, max_new_tokens=20)
                sync_device(USE_DEVICE)
                t1 = time.perf_counter()
                total_time = t1 - t0
                fps = fallback / total_time
                logger.info(f"    -> Time for {fallback} captions: {total_time*1000:.2f} ms")
                logger.info(f"    -> Raw Throughput: {fps:.2f} images/sec")
                logger.info(f"    -> Amortized Latency/image: {(total_time/fallback)*1000:.4f} ms")
                break
            except RuntimeError as e2:
                logger.warning(f"       fallback {fallback} failed: {e2}")
        else:
            logger.error("    -> All fallback batches failed; skipping throughput test.")

    # -------------------------
    # TEST B: REAL-WORLD LATENCY (disk -> preprocess -> clip -> decode)
    # -------------------------
    logger.info("\n[6] REAL-WORLD LATENCY TEST (batch=1)")
    if not os.path.exists(REAL_IMAGE_PATH):
        logger.warning(f"    -> Image not found at {REAL_IMAGE_PATH}. Skipping Test B.")
    else:
        # run multiple iterations and average
        runs = 10
        preprocess_times = []
        clip_times = []
        decode_times = []
        total_times = []
        captions = []

        for i in range(runs):
            sync_device(USE_DEVICE)
            t0 = time.perf_counter()

            # load
            img = Image.open(REAL_IMAGE_PATH).convert("RGB")
            t_load = time.perf_counter()

            # preprocess
            img_tensor = preprocess(img).unsqueeze(0).to(USE_DEVICE)
            sync_device(USE_DEVICE)
            t_pre = time.perf_counter()

            # CLIP encode
            with torch.no_grad():
                t_clip_start = time.perf_counter()
                img_emb = clip_model.encode_image(img_tensor)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                sync_device(USE_DEVICE)
                t_clip_end = time.perf_counter()

                # decode
                t_dec_start = time.perf_counter()
                caption = model.generate(img_emb, tokenizer, max_new_tokens=20)
                sync_device(USE_DEVICE)
                t_dec_end = time.perf_counter()

            t_end = time.perf_counter()

            preprocess_times.append((t_pre - t_load) * 1000.0)   # ms
            clip_times.append((t_clip_end - t_clip_start) * 1000.0)
            decode_times.append((t_dec_end - t_dec_start) * 1000.0)
            total_times.append((t_end - t0) * 1000.0)
            captions.append(safe_to_str(caption))

        # aggregate
        def stats(arr):
            import statistics
            return {
                "mean": statistics.mean(arr),
                "median": statistics.median(arr),
                "min": min(arr),
                "max": max(arr)
            }

        logger.info("    -> Per-stage timings (ms) over {} runs:".format(runs))
        logger.info(f"       Preprocess: {stats(preprocess_times)}")
        logger.info(f"       CLIP encode: {stats(clip_times)}")
        logger.info(f"       Decode (generate): {stats(decode_times)}")
        logger.info(f"       Total E2E: {stats(total_times)}")
        logger.info(f"    -> Example generated caption (last run): {captions[-1]}")

    logger.info("\nDone. Results saved to speed_log.txt")

if __name__ == "__main__":
    main()
