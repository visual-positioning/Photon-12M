import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

# -------------------------------------------------
# Fix import paths
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from benchmarks.mobilecap_modern import build_modern_model
import mobileclip

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
# from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# -------------------------------------------------
# PATHS
# -------------------------------------------------
COCO_IMG_DIR = "/scratch/kalidas_5/Dataset/coco2017/val2017"
COCO_ANN = "/scratch/kalidas_5/Dataset/coco2017/annotations/captions_val2017.json"

TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer", "mobilecap_tokenizer.json")
MODEL_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "nano_ep55.pt")
MOBILECLIP_CKPT = os.path.join(PROJECT_ROOT, "models", "mobileclip_s1.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
print("Loading MobileCLIP image encoder...")
img_model, _, preprocess = mobileclip.create_model_and_transforms(
    "mobileclip_s1", pretrained=MOBILECLIP_CKPT
)
img_model = img_model.to(DEVICE).eval()

print("Loading captioning model...")
model, tokenizer = build_modern_model(TOKENIZER_PATH)
model.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE))
model = model.to(DEVICE).eval()

# -------------------------------------------------
# LOAD GT CAPTIONS
# -------------------------------------------------
with open(COCO_ANN, "r") as f:
    coco = json.load(f)

id_to_fname = {img["id"]: img["file_name"] for img in coco["images"]}

gt = {}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    gt.setdefault(img_id, []).append(ann["caption"])

# -------------------------------------------------
# GENERATE PREDICTIONS
# -------------------------------------------------
preds = {}
gts = {}

idx = 0

with torch.no_grad():
    for img_id, gt_caps in tqdm(gt.items(), desc="Generating captions"):
        fname = id_to_fname[img_id]
        img_path = os.path.join(COCO_IMG_DIR, fname)

        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        emb = img_model.encode_image(image_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)

        caption = model.generate(emb, tokenizer)

        preds[idx] = [caption]
        gts[idx] = gt_caps
        idx += 1

print(f"Evaluating on {len(preds)} images")

# -------------------------------------------------
# METRICS
# -------------------------------------------------
print("\n--- Evaluation Results (COCO val2017) ---")

bleu = Bleu(4)
rouge = Rouge()
# meteor = Meteor()
cider = Cider()
spice = Spice()

bleu_score, _ = bleu.compute_score(gts, preds)
rouge_score, _ = rouge.compute_score(gts, preds)
# meteor_score, _ = meteor.compute_score(gts, preds)
cider_score, _ = cider.compute_score(gts, preds)
spice_score, _ = spice.compute_score(gts, preds)

print(f"BLEU-1 : {bleu_score[0]*100:.2f}")
print(f"BLEU-2 : {bleu_score[1]*100:.2f}")
print(f"BLEU-3 : {bleu_score[2]*100:.2f}")
print(f"BLEU-4 : {bleu_score[3]*100:.2f}")
print(f"ROUGE-L: {rouge_score*100:.2f}")
# print(f"METEOR : {meteor_score*100:.2f}")
print(f"CIDEr  : {cider_score*100:.2f}")
print(f"SPICE  : {spice_score*100:.2f}")
