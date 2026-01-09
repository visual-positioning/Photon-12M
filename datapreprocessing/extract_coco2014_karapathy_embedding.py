"""
extract_coco2014_karapathy_embedding.py

Extract MobileCLIP image embeddings for MSCOCO 2014
using the Karpathy split.

Outputs (saved inside MSCOCO dataset folder):
    MSCOCO2014/embeddings/embeddings_train.pt
    MSCOCO2014/embeddings/embeddings_val.pt
    MSCOCO2014/embeddings/embeddings_test.pt
"""

import os
import json
import torch
import numpy as np
import mobileclip
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# 1. PATH CONFIGURATION (YOUR PATHS)
# ==========================================
COCO_ROOT = "/scratch/kalidas_5/Dataset/MSCOCO2014"
KARPATHY_JSON = "/scratch/kalidas_5/Dataset/karapathy_split/dataset_coco.json"

EMBEDDING_DIR = os.path.join(COCO_ROOT, "embeddings")
os.makedirs(EMBEDDING_DIR, exist_ok=True)

IMAGE_DIRS = {
    "train": "train2014",
    "val": "val2014",
    "test": "val2014",
}

OUTPUT_FILES = {
    "train": os.path.join(EMBEDDING_DIR, "embeddings_train.pt"),
    "val": os.path.join(EMBEDDING_DIR, "embeddings_val.pt"),
    "test": os.path.join(EMBEDDING_DIR, "embeddings_test.pt"),
}

# ==========================================
# 2. MOBILECLIP CONFIG
# ==========================================
MODEL_NAME = "mobileclip_s1"
CHECKPOINT_PATH = "../models/mobileclip_s1.pt"

BATCH_SIZE = 128
NUM_WORKERS = 4

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 3. LOAD KARPATHY SPLITS
# ==========================================
def load_karpathy_splits():
    with open(KARPATHY_JSON, "r") as f:
        data = json.load(f)

    splits = {"train": [], "val": [], "test": []}

    for img in data["images"]:
        split = img["split"]
        if split in splits:
            splits[split].append(img["filename"])

    for k in splits:
        print(f"[INFO] Karpathy {k}: {len(splits[k])} images")

    return splits

# ==========================================
# 4. SAFE DATASET
# ==========================================
class SafeImageDataset(Dataset):
    def __init__(self, img_dir, filenames, transform=None):
        self.img_dir = img_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        path = os.path.join(self.img_dir, fname)

        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, fname, True
        except (UnidentifiedImageError, OSError, Exception):
            dummy = torch.zeros((3, 224, 224))
            return dummy, fname, False

# ==========================================
# 5. EMBEDDING EXTRACTION
# ==========================================
def extract_embeddings(split, model, preprocess, filenames):
    print(f"\n[INFO] Extracting {split} embeddings")

    img_dir = os.path.join(COCO_ROOT, IMAGE_DIRS[split])
    dataset = SafeImageDataset(img_dir, filenames, preprocess)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    embeddings = {}

    with torch.no_grad():
        for images, fnames, valids in tqdm(loader, desc=split):
            mask = valids.bool()
            if not mask.any():
                continue

            images = images[mask].to(device)
            fnames = np.array(fnames)[mask.cpu().numpy()]

            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)

            for f, feat in zip(fnames, feats.cpu()):
                embeddings[f] = feat.half()  # float16

    torch.save(embeddings, OUTPUT_FILES[split])
    print(f"[SUCCESS] Saved {len(embeddings)} → {OUTPUT_FILES[split]}")

# ==========================================
# 6. MAIN
# ==========================================
def main():
    print("[INFO] Device:", device)

    # PyTorch 2.6 MobileCLIP safety patch
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **k, weights_only=False)

    model, _, preprocess = mobileclip.create_model_and_transforms(
        MODEL_NAME, pretrained=CHECKPOINT_PATH
    )

    torch.load = _orig_load

    model = model.to(device)
    model.eval()

    splits = load_karpathy_splits()

    # for split in ["train", "val", "test"]:
    for split in ["test"]:
        extract_embeddings(split, model, preprocess, splits[split])

    print("\n[ALL DONE] Embeddings saved inside MSCOCO2014/embeddings/")

if __name__ == "__main__":
    main()
