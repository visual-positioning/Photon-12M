# train_nano.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Import the model we just created
from mobilecap_modern import build_modern_model

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Paths - ADJUST THESE IF NEEDED
EMBEDDINGS_DIR = "../datapreprocessing/coco_embeddings_s1"      
ANNOTATIONS_PATH = r"C:\Dataset\coco2017\annotations\captions_train2017.json"
VAL_ANNOTATIONS_PATH = r"C:\Dataset\coco2017\annotations\captions_val2017.json"
TOKENIZER_PATH = "../tokenizer/mobilecap_tokenizer.json"
SAVE_DIR = "../checkpoints"

# Hyperparameters
BATCH_SIZE = 512       # Optimized for RTX 4060
LR = 3e-4              # Learning Rate
EPOCHS = 10
WEIGHT_DECAY = 1e-2    # Prevents overfitting
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. DATASET
# ==========================================
class CachedCocoDataset(Dataset):
    def __init__(self, emb_folder, caption_path, tokenizer, split_name="train2017", max_len=32):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        # 1. Load Embeddings into RAM
        print(f"[{split_name}] Loading embeddings...")
        self.emb_map = {}
        split_path = os.path.join(emb_folder, split_name)
        
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} not found. Skipping.")
            return

        chunk_files = sorted([f for f in os.listdir(split_path) if f.endswith('.npz')])
        for cf in tqdm(chunk_files, desc=f"Loading chunks"):
            d = np.load(os.path.join(split_path, cf))
            paths = d['paths']
            embs = d['embeddings']
            for p, e in zip(paths, embs):
                # Ensure float32 for training
                self.emb_map[p] = e.astype(np.float32)
        
        # 2. Load Captions and Match
        print(f"[{split_name}] Matching captions...")
        with open(caption_path, 'r') as f:
            coco = json.load(f)
            
        img_id_map = {img['id']: img['file_name'] for img in coco['images']}
        
        matched = 0
        # Iterate over annotations
        for ann in coco['annotations']:
            fname = img_id_map.get(ann['image_id'])
            if fname in self.emb_map:
                self.data.append((fname, ann['caption']))
                matched += 1
                
        print(f"[{split_name}] Ready. {matched} pairs loaded.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname, caption = self.data[idx]
        
        # Get Embedding
        img_emb = torch.from_numpy(self.emb_map[fname])
        
        # Tokenize
        # We use simple encoding, we handle special tokens manually if needed, 
        # but our tokenizer adds CLS/SEP automatically via post-processing if configured.
        # Just to be safe, let's encode normally.
        tokens = self.tokenizer.encode(caption)
        
        # Truncate
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        
        # Pad
        pad_id = self.tokenizer.pad_token_id
        pad_needed = self.max_len - len(tokens)
        input_ids = tokens + [pad_id] * pad_needed
        
        return img_emb, torch.tensor(input_ids, dtype=torch.long)

# ==========================================
# 3. UTILITIES
# ==========================================
def plot_history(train_losses, val_losses, epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    if val_losses and any(val_losses):
        plt.plot(val_losses, label='Val Loss')
    plt.title(f"Training Progress - Epoch {epoch+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "loss_plot.png"))
    plt.close()

# ==========================================
# 4. MAIN LOOP
# ==========================================
def main():
    # 1. Build Model
    model, tokenizer = build_modern_model(TOKENIZER_PATH)
    if model is None: return
    
    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 2. Load Data
    train_ds = CachedCocoDataset(EMBEDDINGS_DIR, ANNOTATIONS_PATH, tokenizer, split_name="train2017")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    # Optional Val
    val_ds = CachedCocoDataset(EMBEDDINGS_DIR, VAL_ANNOTATIONS_PATH, tokenizer, split_name="val2017")
    has_val = len(val_ds) > 0
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False) if has_val else None

    # 3. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler() # Faster training on RTX cards
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    history_train = []
    history_val = []

    print("\n>>> STARTING TRAINING")
    
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        total_loss = 0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for img_emb, input_ids in pbar:
            img_emb, input_ids = img_emb.to(DEVICE), input_ids.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits = model(img_emb, input_ids)
                
                # Shift labels: Input "The cat" -> Predict "cat sat"
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            steps += 1
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
        avg_train = total_loss / steps
        history_train.append(avg_train)

        # --- VALIDATION ---
        avg_val = 0
        if has_val:
            model.eval()
            val_loss_accum = 0
            val_steps = 0
            with torch.no_grad():
                for img_emb, input_ids in val_loader:
                    img_emb, input_ids = img_emb.to(DEVICE), input_ids.to(DEVICE)
                    logits = model(img_emb, input_ids)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    val_loss_accum += loss.item()
                    val_steps += 1
            avg_val = val_loss_accum / val_steps
            history_val.append(avg_val)

        print(f"Epoch {epoch+1} Done. Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        # --- PLOT & SAVE ---
        plot_history(history_train, history_val, epoch)
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"nano_ep{epoch+1}.pt"))

        # --- GENERATE RANDOM SAMPLES (MONITORING) ---
        print("\n--- Model Thoughts ---")
        # 5 samples from Train
        for _ in range(3):
            idx = random.randint(0, len(train_ds)-1)
            emb, _ = train_ds[idx]
            cap = model.generate(emb.unsqueeze(0).to(DEVICE), tokenizer)
            print(f"[Train] {train_ds.data[idx][0]}: {cap}")

        # 5 samples from Val
        if has_val:
            for _ in range(3):
                idx = random.randint(0, len(val_ds)-1)
                emb, _ = val_ds[idx]
                cap = model.generate(emb.unsqueeze(0).to(DEVICE), tokenizer)
                print(f"[Val]   {val_ds.data[idx][0]}: {cap}")
        print("----------------------\n")

if __name__ == "__main__":
    main()