import torch
import json
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import mobileclip
from mobilecap_modern import build_modern_model

# ==========================================
# 1. CONFIGURATION
# ==========================================
TEST_IMG_DIR = r"C:\Dataset\coco2017\test2017" 
TEST_INFO_FILE = r"C:\Dataset\coco2017\annotations\image_info_test2017.json"

# Model Paths
CHECKPOINT_PATH = "../checkpoints/nano_ep5.pt" 
TOKENIZER_PATH = "../tokenizer/mobilecap_tokenizer.json"

OUTPUT_JSON = "captions_test2017_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🚀 BATCH SETTINGS
BATCH_SIZE = 32         # Adjusted for safety
NUM_WORKERS = 0         # Set to 0 for Windows stability (avoid multiprocessing issues)

# ==========================================
# 2. DATASET CLASS
# ==========================================
class CocoTestDataset(Dataset):
    def __init__(self, img_dir, info_file, transform):
        self.img_dir = img_dir
        self.transform = transform
        
        print(f"Loading metadata from {info_file}...")
        with open(info_file, 'r') as f:
            data = json.load(f)
        self.images = data['images']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        image_id = info['id']
        file_name = info['file_name']
        img_path = os.path.join(self.img_dir, file_name)

        try:
            # Load and Transform
            image = Image.open(img_path).convert("RGB")
            tensor = self.transform(image)
            return tensor, image_id
        except Exception as e:
            print(f"Warning: Error loading {file_name}: {e}")
            # Return dummy if failed
            dummy = torch.zeros((3, 224, 224))
            return dummy, image_id

# ==========================================
# 3. BATCH GENERATOR LOGIC
# ==========================================
def batch_generate(model, img_embs, tokenizer, max_tokens=20):
    """
    Manually runs generation for a batch.
    Does NOT use model.generate() to avoid the single-item logic inside it.
    """
    B = img_embs.shape[0]
    device = img_embs.device
    
    # Start tokens [B, 1]
    start_token = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
    input_ids = torch.tensor([[start_token]] * B, device=device)
    
    # Simple Greedy Loop
    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(img_embs, input_ids)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    # Decode batch
    captions = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    return captions

# ==========================================
# 4. MAIN LOOP
# ==========================================
def main():
    print(f"--- 🚀 GENERATING SUBMISSION (BATCH: {BATCH_SIZE}) ---")

    # 1. Load Models
    print("1. Loading Models...")
    model, tokenizer = build_modern_model(TOKENIZER_PATH)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    
    # NOTE: REMOVED torch.compile() because it fails on Windows without Triton.
    # It is not strictly necessary for inference, just a speedup.
    
    clip_model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s1', pretrained='../models/mobileclip_s1.pt')
    clip_model = clip_model.to(DEVICE).eval()

    # 2. Setup DataLoader
    print("2. Setting up DataLoader...")
    dataset = CocoTestDataset(TEST_IMG_DIR, TEST_INFO_FILE, preprocess)
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, # 0 is safer on Windows
        pin_memory=True
    )
    
    print(f"   Total Images: {len(dataset)}")

    # 3. Inference Loop
    results = []
    
    print("3. Running Batch Inference...")
    
    with torch.no_grad():
        for imgs, img_ids in tqdm(loader):
            imgs = imgs.to(DEVICE)
            
            # A. Encode (Vision)
            img_embs = clip_model.encode_image(imgs)
            img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
            
            # B. Decode (Language)
            captions = batch_generate(model, img_embs, tokenizer, max_tokens=20)
            
            # C. Store Results
            for i, cap in enumerate(captions):
                # Clean up string
                clean_cap = cap.replace("\n", "").strip()
                
                results.append({
                    "image_id": img_ids[i].item(),
                    "caption": clean_cap
                })

    # 4. Save
    print(f"4. Saving {len(results)} captions to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f)
        
    print("✅ DONE! Ready for CodaLab upload.")

if __name__ == "__main__":
    main()