# evaluate_coco2014_valtest.py
# =========================================================
# Evaluation on MSCOCO 2014 Karpathy TEST split
# Metrics: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, SPICE
# =========================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import os
import json
import torch
from tqdm import tqdm

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# ---------------------------------------------------------
# PATHS (ADJUST ONLY IF NEEDED)
# ---------------------------------------------------------
EMBEDDINGS_PATH = "/scratch/kalidas_5/Dataset/MSCOCO2014/embeddings/embeddings_test.pt"
KARPATHY_JSON = "/scratch/kalidas_5/Dataset/karapathy_split/dataset_coco.json"
CHECKPOINT = "/scratch/kalidas_5/Photon-12M/checkpoints/coco2014/nano_best.pt"
TOKENIZER_PATH = "/scratch/kalidas_5/Photon-12M/tokenizer/coco2014_tokenizer.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEAM_SIZE = 7


# ---------------------------------------------------------
# IMPORT MODEL
# ---------------------------------------------------------
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, PROJECT_ROOT)

from architecture.mobilecap_modern_beam import build_modern_model

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model, tokenizer = build_modern_model(TOKENIZER_PATH)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------------------------------------------------------
# LOAD EMBEDDINGS
# ---------------------------------------------------------
embeddings = torch.load(EMBEDDINGS_PATH, weights_only=False)

# ---------------------------------------------------------
# LOAD KARPATHY TEST CAPTIONS
# ---------------------------------------------------------
with open(KARPATHY_JSON, "r") as f:
    karpathy = json.load(f)

test_images = [img for img in karpathy["images"] if img["split"] == "test"]

print(f"[INFO] Karpathy test images: {len(test_images)}")

# ---------------------------------------------------------
# GENERATE PREDICTIONS
# ---------------------------------------------------------
gts = {}   # ground truth captions
res = {}   # generated captions

print("[INFO] Generating captions...")

for idx, img in enumerate(tqdm(test_images)):
    fname = img["filename"]

    if fname not in embeddings:
        continue

    emb = embeddings[fname].unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        caption = model.generate(
            emb,
            tokenizer,
            beam_size=BEAM_SIZE
        )


    gts[idx] = [s["raw"] for s in img["sentences"]]
    res[idx] = [caption]


# ---------------------------------------------------------
# COMPUTE METRICS
# ---------------------------------------------------------
print("\n[INFO] Computing metrics...")

scorers = [
    (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
    # (Meteor(), "METEOR"),  # METEOR is disabled due to Java dependency issues
    (Rouge(), "ROUGE-L"),
    (Cider(), "CIDEr"),
    (Spice(), "SPICE"),
]

final_scores = {}

for scorer, method in scorers:
    score, scores = scorer.compute_score(gts, res)
    if isinstance(method, list):
        for m, s in zip(method, score):
            final_scores[m] = s
    else:
        final_scores[method] = score

# ---------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------
print("\n========== COCO KARPATHY TEST RESULTS (%) ==========")
for k, v in final_scores.items():
    print(f"{k:8s}: {v * 100:.2f}%")
print("===================================================")

