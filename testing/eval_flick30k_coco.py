import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


import os
import json
import numpy as np
import torch
from tqdm import tqdm

from benchmarks.mobilecap_modern import build_modern_model
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# -----------------------------
# PATHS
# -----------------------------
EMB_ROOT = "../../Dataset/flickr30k_images/embedding/test"
KARPATHY_JSON = "../../Dataset/flickr30k_images/dataset_flickr30k.json"

TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer", "mobilecap_tokenizer.json")
MODEL_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "nano_ep55.pt")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD MODEL
# -----------------------------
model, tokenizer = build_modern_model(TOKENIZER_PATH)

if model is None or tokenizer is None:
    raise RuntimeError("Failed to load model/tokenizer. Check paths.")

model.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# LOAD GT CAPTIONS (TEST ONLY)
# -----------------------------
with open(KARPATHY_JSON) as f:
    data = json.load(f)

gt_captions = {}
for img in data["images"]:
    if img["split"] != "test":
        continue
    gt_captions[img["filename"]] = [s["raw"] for s in img["sentences"]]

# -----------------------------
# GENERATE PREDICTIONS
# -----------------------------
preds = {}
gts = {}

chunk_files = sorted(f for f in os.listdir(EMB_ROOT) if f.endswith(".npz"))

idx = 0
for cf in tqdm(chunk_files, desc="Generating captions"):
    d = np.load(os.path.join(EMB_ROOT, cf))
    embeddings = d["embeddings"]
    paths = d["paths"]

    for emb, fname in zip(embeddings, paths):
        if fname not in gt_captions:
            continue

        emb = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        caption = model.generate(emb, tokenizer)

        preds[idx] = [caption]
        gts[idx] = gt_captions[fname]
        idx += 1

# -----------------------------
# METRIC COMPUTATION
# -----------------------------
print("\n--- Evaluation Results ---")

bleu = Bleu(4)
rouge = Rouge()
meteor = Meteor()
cider = Cider()
spice = Spice()

bleu_score, _ = bleu.compute_score(gts, preds)
rouge_score, _ = rouge.compute_score(gts, preds)
meteor_score, _ = meteor.compute_score(gts, preds)
cider_score, _ = cider.compute_score(gts, preds)
spice_score, _ = spice.compute_score(gts, preds)

print(f"BLEU-1: {bleu_score[0] * 100:.2f}%")
print(f"BLEU-2: {bleu_score[1] * 100:.2f}%")
print(f"BLEU-3: {bleu_score[2] * 100:.2f}%")
print(f"BLEU-4: {bleu_score[3] * 100:.2f}%")
print(f"ROUGE-L: {rouge_score * 100:.2f}%")
print(f"METEOR: {meteor_score * 100:.2f}%")
print(f"CIDEr: {cider_score * 100:.2f}%")
print(f"SPICE: {spice_score * 100:.2f}%")

