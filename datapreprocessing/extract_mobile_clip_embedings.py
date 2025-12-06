import os
import torch
import numpy as np
import mobileclip
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Adjust this path to your COCO root
COCO_ROOT = r"C:\Dataset\coco2017" 

# Output directory for embeddings
OUT_DIR = "coco_embeddings_s1"

# Hyperparameters
MODEL_NAME = 'mobileclip_s1'
CHECKPOINT_PATH = '../models/mobileclip_s1.pt' # Ensure this file exists
BATCH_SIZE = 128                     # 256 or 512 if you have a big GPU (RTX 3090/4090)
NUM_WORKERS = 2                      # Number of CPU cores for image loading
SAVE_CHUNK_SIZE = 1000              # Save to disk every 10k images (Safety)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. ROBUST DATASET CLASS
# ==========================================
class SafeImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(img_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        print(f"Found {len(self.image_files)} images in {img_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        path = os.path.join(self.img_dir, filename)
        
        try:
            # 1. Open Image
            image = Image.open(path).convert("RGB")
            
            # 2. Transform
            if self.transform:
                image = self.transform(image)
            
            # Return valid data
            return image, filename, True

        except (UnidentifiedImageError, OSError, Exception) as e:
            # SAFETY: Return dummy data if corrupt, flag as invalid
            print(f"WARNING: Skipping corrupt image {filename}: {e}")
            # Return a dummy tensor of correct shape to keep batching valid
            dummy = torch.zeros((3, 224, 224)) 
            return dummy, filename, False

# ==========================================
# 3. EXTRACTION ENGINE
# ==========================================
def extract_split(split_name, img_dir, model, preprocess):
    print(f"\n--- Processing {split_name} ---")
    
    # Setup Data
    dataset = SafeImageDataset(img_dir, transform=preprocess)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # Storage buffers
    all_embeddings = []
    all_paths = []
    
    # Create split output dir
    split_out_dir = os.path.join(OUT_DIR, split_name)
    os.makedirs(split_out_dir, exist_ok=True)
    
    chunk_counter = 0
    total_processed = 0

    # Inference Loop
    with torch.no_grad():
        for images, filenames, valids in tqdm(loader, desc=f"Extracting {split_name}"):
            
            # Filter out corrupt images from this batch
            # valids is a boolean tensor indicating which images loaded successfully
            valid_mask = valids.bool()
            
            if not valid_mask.any():
                continue # Skip if whole batch is corrupt
                
            clean_images = images[valid_mask].to(device)
            clean_filenames = np.array(filenames)[valid_mask.cpu().numpy()]
            
            # Extract features
            features = model.encode_image(clean_images)
            features = features / features.norm(dim=-1, keepdim=True)
            
            # Move to CPU to save memory
            all_embeddings.append(features.cpu().numpy().astype(np.float16)) # float16 saves 50% disk space
            all_paths.extend(clean_filenames)
            
            total_processed += len(clean_filenames)

            # SAFETY SAVE: If buffer gets too big, dump to disk
            if total_processed >= (chunk_counter + 1) * SAVE_CHUNK_SIZE:
                save_chunk(split_out_dir, chunk_counter, all_embeddings, all_paths)
                # Reset buffers
                all_embeddings = []
                all_paths = []
                chunk_counter += 1

    # Save remaining data
    if len(all_paths) > 0:
        save_chunk(split_out_dir, chunk_counter, all_embeddings, all_paths)

    print(f"Done. {split_name} processed successfully.")

def save_chunk(folder, chunk_id, embeddings_list, paths_list):
    """Saves a partial result to disk"""
    if not embeddings_list:
        return

    # Concatenate list of arrays into one big array
    emb_array = np.concatenate(embeddings_list, axis=0)
    path_array = np.array(paths_list)
    
    filename = os.path.join(folder, f"chunk_{chunk_id:03d}")
    
    np.savez_compressed(
        filename, 
        embeddings=emb_array, 
        paths=path_array
    )
    print(f" [Safety] Saved chunk {chunk_id} ({len(path_array)} items) to {filename}.npz")

# ==========================================
# 4. MAIN
# ==========================================
def main():
    # 1. Load Model
    print("Loading MobileCLIP model...")
    
    # Patch for PyTorch 2.6 safety
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **k, weights_only=False) if 'weights_only' not in k else _orig_load(*a, **k)
    
    model, _, preprocess = mobileclip.create_model_and_transforms(MODEL_NAME, pretrained=CHECKPOINT_PATH)
    torch.load = _orig_load
    
    model = model.to(device)
    model.eval()

    # 2. Process Validation Set (Smaller, good for testing)
    val_dir = os.path.join(COCO_ROOT, "val2017")
    if os.path.exists(val_dir):
        extract_split("val2017", val_dir, model, preprocess)
    else:
        print(f"Skipping val2017 (Folder not found: {val_dir})")

    # 3. Process Train Set (Large)
    train_dir = os.path.join(COCO_ROOT, "train2017")
    if os.path.exists(train_dir):
        extract_split("train2017", train_dir, model, preprocess)
    else:
        print(f"Skipping train2017 (Folder not found: {train_dir})")

    # 4. Final Merger (Optional helper to create one big file)
    print("\nExtraction complete. You have chunked .npz files in:", OUT_DIR)
    print("You can load them individually, or I can provide a script to merge them if you prefer.")

if __name__ == "__main__":
    main()