import os
import json
import torch
import numpy as np
import mobileclip
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
FLICKR_ROOT = "../Dataset/flickr30k_images"

IMAGE_DIR = os.path.join(FLICKR_ROOT, "flickr30k_images")
SPLIT_JSON = os.path.join(FLICKR_ROOT, "dataset_flickr30k.json")

# OUT_DIR = "datapreprocessing/flickr30k_embeddings_s1"
OUT_DIR = os.path.join(FLICKR_ROOT, "embedding")

MODEL_NAME = "mobileclip_s1"
CHECKPOINT_PATH = "models/mobileclip_s1.pt"

BATCH_SIZE = 128
NUM_WORKERS = 2
SAVE_CHUNK_SIZE = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. LOAD KARPATHY SPLITS
# ==========================================
def load_karpathy_splits(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    splits = {"train": [], "val": [], "test": []}
    for item in data["images"]:
        split = item["split"]
        if split in splits:
            splits[split].append(item["filename"])

    return splits

# ==========================================
# 3. SAFE DATASET
# ==========================================
class SafeImageDataset(Dataset):
    def __init__(self, img_dir, image_list, transform=None):
        self.img_dir = img_dir
        self.image_list = image_list
        self.transform = transform
        print(f"Loaded {len(image_list)} images")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        filename = self.image_list[idx]
        path = os.path.join(self.img_dir, filename)

        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, filename, True

        except (UnidentifiedImageError, OSError, Exception) as e:
            print(f"WARNING: Corrupt image {filename}: {e}")
            dummy = torch.zeros((3, 224, 224))
            return dummy, filename, False

# ==========================================
# 4. SAVE CHUNKS
# ==========================================
def save_chunk(folder, chunk_id, embeddings_list, paths_list):
    if not embeddings_list:
        return

    emb_array = np.concatenate(embeddings_list, axis=0)
    path_array = np.array(paths_list)

    out_file = os.path.join(folder, f"chunk_{chunk_id:03d}.npz")
    np.savez_compressed(out_file, embeddings=emb_array, paths=path_array)

    print(f" [Saved] {out_file} ({len(path_array)} items)")

# ==========================================
# 5. EXTRACTION ENGINE
# ==========================================
def extract_split(split_name, image_list, model, preprocess):
    print(f"\n--- Extracting {split_name} ---")

    dataset = SafeImageDataset(IMAGE_DIR, image_list, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    out_split_dir = os.path.join(OUT_DIR, split_name)
    os.makedirs(out_split_dir, exist_ok=True)

    all_embeddings, all_paths = [], []
    chunk_id = 0
    total = 0

    with torch.no_grad():
        for images, filenames, valids in tqdm(loader, desc=split_name):
            valid_mask = valids.bool()
            if not valid_mask.any():
                continue

            images = images[valid_mask].to(device)
            filenames = np.array(filenames)[valid_mask.cpu().numpy()]

            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)

            all_embeddings.append(feats.cpu().numpy().astype(np.float16))
            all_paths.extend(filenames)
            total += len(filenames)

            if total >= (chunk_id + 1) * SAVE_CHUNK_SIZE:
                save_chunk(out_split_dir, chunk_id, all_embeddings, all_paths)
                all_embeddings, all_paths = [], []
                chunk_id += 1

    if all_paths:
        save_chunk(out_split_dir, chunk_id, all_embeddings, all_paths)

    print(f"Done {split_name}")

# ==========================================
# 6. MAIN
# ==========================================
def main():
    print("Loading MobileCLIP...")

    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **k, weights_only=False) \
        if 'weights_only' not in k else _orig_load(*a, **k)

    model, _, preprocess = mobileclip.create_model_and_transforms(
        MODEL_NAME, pretrained=CHECKPOINT_PATH
    )
    torch.load = _orig_load

    model = model.to(device)
    model.eval()

    splits = load_karpathy_splits(SPLIT_JSON)

    for split in ["train", "val", "test"]:
        extract_split(split, splits[split], model, preprocess)

    print("\nFlickr30k Karpathy embedding extraction complete.")
    print("Saved at:", OUT_DIR)

if __name__ == "__main__":
    main()
