import os
import sys
import json
import torch
from tqdm import tqdm
from PIL import Image

# ------------------------------------------------------------
# Add PROJECT ROOT to PYTHONPATH
# ------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------
# Imports from your repo
# ------------------------------------------------------------
import mobileclip
from benchmarks.mobilecap_modern import build_modern_model

# ------------------------------------------------------------
# PATHS (MATCH YOUR SETUP)
# ------------------------------------------------------------
NOCAPS_IMG_DIR = "/scratch/kalidas_5/Dataset/nocaps/test"
NOCAPS_TEST_JSON = "/scratch/kalidas_5/Dataset/nocaps/nocaps_test_public.json"

TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer", "mobilecap_tokenizer.json")
MODEL_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "nano_ep55.pt")
MOBILECLIP_CKPT = os.path.join(PROJECT_ROOT, "models", "mobileclip_s1.pt")

OUTPUT_JSON = "nocaps_test_predictions.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Load MobileCLIP image encoder
# ------------------------------------------------------------
print("Loading MobileCLIP image encoder...")
image_encoder, _, preprocess = mobileclip.create_model_and_transforms(
    "mobileclip_s1",
    pretrained=MOBILECLIP_CKPT
)
image_encoder = image_encoder.to(DEVICE)
image_encoder.eval()

# ------------------------------------------------------------
# Load captioning model
# ------------------------------------------------------------
print("Loading captioning model...")
model, tokenizer = build_modern_model(TOKENIZER_PATH)

if model is None or tokenizer is None:
    raise RuntimeError("Failed to load model or tokenizer.")

state = torch.load(MODEL_CKPT, map_location=DEVICE)
model.load_state_dict(state)
model = model.to(DEVICE)
model.eval()

# ------------------------------------------------------------
# Load NoCaps TEST metadata
# ------------------------------------------------------------
print("Loading NoCaps TEST public metadata...")
with open(NOCAPS_TEST_JSON, "r") as f:
    nocaps_images = json.load(f)["images"]

print(f"Total NoCaps test images: {len(nocaps_images)}")

# ------------------------------------------------------------
# Caption generation
# ------------------------------------------------------------
predictions = []
missing = 0

with torch.no_grad():
    for sample in tqdm(nocaps_images, desc="Generating NoCaps captions"):

        # IMPORTANT: two different IDs
        eval_image_id = sample["id"]                 # integer → EvalAI
        open_images_id = sample["open_images_id"]    # string → filename

        img_path = os.path.join(
            NOCAPS_IMG_DIR, f"{open_images_id}.jpg"
        )

        if not os.path.exists(img_path):
            missing += 1
            print("MISSING IMAGE:", open_images_id)
            continue

        image = Image.open(img_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        # MobileCLIP forward
        img_emb = image_encoder(image_tensor)
        if isinstance(img_emb, (tuple, list)):
            img_emb = img_emb[0]

        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)


        # Caption generation
        caption = model.generate(img_emb, tokenizer)

        # Ensure string output
        if not isinstance(caption, str):
            caption = tokenizer.decode(caption, skip_special_tokens=True)

        predictions.append({
            "image_id": int(eval_image_id),
            "caption": caption
        })

# ------------------------------------------------------------
# Save EvalAI submission JSON
# ------------------------------------------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(predictions, f)

print("\n================ SUMMARY ================")
print(f"Generated captions : {len(predictions)}")
print(f"Missing images     : {missing}")
print(f"Saved to           : {OUTPUT_JSON}")
print("========================================")
print("Next step: zip this file and submit to EvalAI.")
