import torch
import time
import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

# Metric imports
from torchmetrics.text import BLEUScore, ROUGEScore
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# --- CRITICAL IMPORTS FOR MANUAL SCORER SELECTION ---
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice  <-- WE ARE DISABLING THIS

import mobileclip
from mobilecap_modern import build_modern_model

# ==========================================
# 1. CONFIGURATION
# ==========================================
CHECKPOINT_PATH = "../checkpoints/nano_ep5.pt" 
TOKENIZER_PATH = "../tokenizer/mobilecap_tokenizer.json"
TEST_IMG_DIR = r"C:\Dataset\coco2017\val2017" 
TEST_ANN_FILE = r"C:\Dataset\coco2017\annotations\captions_val2017.json"
MAX_SAMPLES = None  # Set to None for full run, 100 is good for quick accurate stats
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. FLOPs ESTIMATOR
# ==========================================
def estimate_flops(model, context_len=20):
    total_params = sum(p.numel() for p in model.parameters())
    return 2 * total_params * context_len

# ==========================================
# 3. MAIN EVALUATION
# ==========================================
def main():
    print(f"--- 🚀 STARTING FINAL AUDIT ---")
    
    # 1. Load Models
    model, tokenizer = build_modern_model(TOKENIZER_PATH) 
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    
    clip_model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s1', pretrained='../models/mobileclip_s1.pt')
    clip_model = clip_model.to(DEVICE).eval()

    # 2. Setup Data
    coco = COCO(TEST_ANN_FILE)
    img_ids = coco.getImgIds()
    if MAX_SAMPLES: img_ids = img_ids[:MAX_SAMPLES]
    
    print(f"Evaluating {len(img_ids)} samples...")

    # Storage
    coco_results = []
    generated_caps = []
    reference_caps = []
    latencies = []

    # 3. Inference Loop
    # Warmup GPU (fixes the slow first epoch timing issue)
    print("Warming up GPU...")
    dummy_img = torch.randn(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        d_emb = clip_model.encode_image(dummy_img)
        model.generate(d_emb, tokenizer, max_new_tokens=5)
    
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    print("Running Inference...")
    for img_id in tqdm(img_ids):
        try:
            # Load Image
            info = coco.loadImgs(img_id)[0]
            path = os.path.join(TEST_IMG_DIR, info['file_name'])
            image = Image.open(path).convert("RGB")
            tensor = preprocess(image).unsqueeze(0).to(DEVICE)
            
            # Timing
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.time()
            
            with torch.no_grad():
                emb = clip_model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                
                caption = model.generate(emb, tokenizer, max_new_tokens=20)
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            latencies.append((time.time() - t0) * 1000)

            # Store
            coco_results.append({"image_id": img_id, "caption": caption})
            
            # Get Ground Truth
            ann_ids = coco.getAnnIds(imgIds=img_id)
            refs = [a['caption'] for a in coco.loadAnns(ann_ids)]
            generated_caps.append(caption)
            reference_caps.append(refs)

        except Exception as e:
            print(f"Error on {img_id}: {e}")

    # 4. Metrics
    print("\nCalculating Scores...")
    
    # A. TorchMetrics (Backup)
    tm_bleu = BLEUScore(n_gram=4)(generated_caps, reference_caps)
    
    # B. COCO Eval (The Gold Standard)
    coco_metrics = {}
    try:
        with open("preds.json", "w") as f: json.dump(coco_results, f)
        coco_eval = COCOEvalCap(coco, coco.loadRes("preds.json"))
        coco_eval.params['image_id'] = [x['image_id'] for x in coco_results]

        # --- EXPLICITLY DEFINE SCORERS (NO SPICE) ---
        coco_eval.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
            # SPICE removed to prevent Java 21 crash
        ]
        # --------------------------------------------

        coco_eval.evaluate()
        coco_metrics = {k: v for k, v in coco_eval.eval.items()}
    except Exception as e:
        print(f"COCO Eval Error: {e}")

    # 5. Report
    avg_lat = np.mean(latencies)
    flops = estimate_flops(model) / 1e9
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    
    print("\n" + "="*45)
    print("  FINAL SCORECARD")
    print("="*45)
    print(f"MODEL      : Nano-LlaMA (12M Params)")
    print("-" * 45)
    print(f"SPEED      : {avg_lat:.2f} ms/image ({1000/avg_lat:.2f} FPS)")
    print(f"EFFICIENCY : {flops:.4f} GFLOPs")
    print("-" * 45)
    print(f"QUALITY (COCO Standard):")
    if 'CIDEr' in coco_metrics:
        print(f"  CIDEr    : {coco_metrics['CIDEr']:.4f}  <-- MAIN METRIC (Aim > 0.8)")
    print(f"  BLEU-4   : {coco_metrics.get('Bleu_4', 0):.4f}")
    print(f"  ROUGE-L  : {coco_metrics.get('ROUGE_L', 0):.4f}")
    print(f"  METEOR   : {coco_metrics.get('METEOR', 0):.4f}")
    print("="*45)

if __name__ == "__main__":
    main()