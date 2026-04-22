import os
import argparse
import yaml
import logging
import json
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Environment settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Import your model builder ---
# Ensure this import matches your folder structure!
from architecture.mobilecap_ablation import build_modern_model

# ==========================================
# 1. SETUP & UTILS
# ==========================================

def load_config(yaml_path, idx):
    with open(yaml_path, 'r') as f:
        configs = yaml.safe_load(f)
    if idx >= len(configs):
        raise IndexError(f"Config index {idx} out of range. Only {len(configs)} configs found.")
    print(f"Loaded Configuration [{idx}]: {configs[idx]['experiment_name']}")
    return configs[idx]

def setup_experiment_dir(experiment_name, base_dir="./checkpoints"):
    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def setup_logger(exp_dir):
    log_file = os.path.join(exp_dir, "loss.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Logging started. Saving to {log_file}")

def plot_history(train_losses, val_losses, exp_dir, epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    if val_losses and any(v > 0 for v in val_losses):
        plt.plot(val_losses, label='Val Loss')
    plt.title(f"Training Progress - Epoch {epoch+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(exp_dir, "plot.png"))
    plt.close()

# ==========================================
# 2. DATASET CLASS (UPDATED FOR RESTVAL)
# ==========================================
class KarpathyDataset(Dataset):
    def __init__(self, emb_file_paths, karpathy_json, tokenizer, split_name="train", max_len=32):
        """
        emb_file_paths: List of paths to .pt files (e.g. [train.pt, val.pt])
                        We need BOTH for training because 'restval' images are in val.pt
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        # 1. Load Embeddings (Merge dictionaries from multiple files)
        self.emb_map = {}
        
        # Ensure input is a list
        if isinstance(emb_file_paths, str):
            emb_file_paths = [emb_file_paths]
            
        for path in emb_file_paths:
            logging.info(f"[{split_name}] Loading Embeddings from: {path}")
            if os.path.exists(path):
                # weights_only=False required for numpy compat
                partial_map = torch.load(path, map_location="cpu", weights_only=False)
                self.emb_map.update(partial_map)
                logging.info(f"   -> Found {len(partial_map)} entries.")
            else:
                logging.warning(f"   -> File not found: {path}")
        
        logging.info(f"[{split_name}] Total Embeddings Available: {len(self.emb_map)}")

        # 2. Load Karpathy JSON
        logging.info(f"[{split_name}] Loading Captions from: {karpathy_json}")
        with open(karpathy_json, 'r') as f:
            content = json.load(f)
            
        # 3. Define Splits
        # CRITICAL: If 'train', we want 'train' AND 'restval'
        if split_name == 'train':
            target_splits = ['train', 'restval']
        elif split_name == 'val':
            target_splits = ['val']
        elif split_name == 'test':
            target_splits = ['test']
        else:
            target_splits = [split_name]
            
        logging.info(f"[{split_name}] Filtering for splits: {target_splits}")

        # 4. Parse JSON
        matched_count = 0
        missing_count = 0
        
        for img in tqdm(content['images'], desc=f"Parsing {split_name}"):
            if img['split'] in target_splits:
                fname = img['filename']
                
                # Check if we have the embedding
                if fname in self.emb_map:
                    for sent in img['sentences']:
                        caption = sent['raw']
                        self.data.append((fname, caption))
                    matched_count += 1
                else:
                    missing_count += 1

        logging.info(f"[{split_name}] Ready. {len(self.data)} pairs. (Images Matched: {matched_count}, Missing: {missing_count})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname, caption = self.data[idx]
        
        # Get Embedding (Float32 for training stability)
        img_emb = self.emb_map[fname].float() 
        
        # Tokenize
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
# 3. MAIN TRAINING LOOP
# ==========================================
def run_training(config_idx, config_file="config.yaml"):
    # 1. Load Config
    cfg = load_config(config_file, config_idx)
    exp_dir = setup_experiment_dir(cfg['experiment_name'])
    setup_logger(exp_dir)
    logging.info(f"Configuration: {json.dumps(cfg, indent=2)}")

    # 2. Build Model
    model_cfg = {
        "d_model": cfg['d_model'],
        "num_heads": cfg['num_heads'],
        "ffn_hidden": cfg['ffn_hidden'],
        "num_layers": cfg['num_layers'],
        "max_length": cfg['max_length'],
        "num_img_tokens": cfg.get('num_img_tokens', 8),
        "conditioning_strategy": cfg.get('conditioning_strategy', 'prefix'),
        "device": cfg['device']
    }
    
    model, tokenizer = build_modern_model(tokenizer_path=cfg['tokenizer_path'], override_cfg=model_cfg)
    if model is None: return
    device = cfg['device']
    model = model.to(device)
    logging.info(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 3. Data Setup
    user_home = os.path.expanduser("~")
    def resolve_path(p):
        if p.startswith("~"): return os.path.expanduser(p)
        if os.path.isabs(p): return p
        return os.path.join(user_home, p)

    emb_dir_path = resolve_path(cfg['embeddings_dir'])
    karpathy_path = resolve_path(cfg['karpathy_json_path'])
    
    # Path to embeddings
    train_pt_path = os.path.join(emb_dir_path, "embeddings_train.pt")
    val_pt_path = os.path.join(emb_dir_path, "embeddings_val.pt")

    # --- TRAIN DATASET (Combined Sources) ---
    # We pass BOTH files. The dataset class will merge them and filter for 'train' + 'restval'
    logging.info("--- Preparing TRAINING Set (Train + RestVal) ---")
    train_ds = KarpathyDataset(
        emb_file_paths=[train_pt_path, val_pt_path], 
        karpathy_json=karpathy_path, 
        tokenizer=tokenizer, 
        split_name="train", # This now triggers [train, restval] logic
        max_len=cfg['max_length']
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    # --- VAL DATASET ---
    # We only need val_pt for validation split
    logging.info("--- Preparing VALIDATION Set ---")
    if os.path.exists(val_pt_path):
        val_ds = KarpathyDataset(
            emb_file_paths=val_pt_path, 
            karpathy_json=karpathy_path, 
            tokenizer=tokenizer, 
            split_name="val",
            max_len=cfg['max_length']
        )
        val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)
    else:
        val_loader = None
        logging.warning("Validation embeddings not found. Skipping validation.")

    # 4. Optimizer & Scaler
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Mixed Precision Logic
    use_amp = cfg.get('mixed_precision', True)
    # Use standard cuda scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    logging.info(f"Mixed Precision Enabled: {use_amp}")
    logging.info(">>> STARTING TRAINING")

    history_train, history_val = [], []

    for epoch in range(cfg['epochs']):
        # --- TRAIN ---
        model.train()
        total_loss, steps = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        
        for img_emb, input_ids in pbar:
            img_emb, input_ids = img_emb.to(device), input_ids.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(img_emb, input_ids)
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
        
        # --- VAL ---
        avg_val = 0
        if val_loader:
            model.eval()
            val_loss, val_steps = 0, 0
            with torch.no_grad():
                for img_emb, input_ids in val_loader:
                    img_emb, input_ids = img_emb.to(device), input_ids.to(device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = model(img_emb, input_ids)
                        loss = loss_fn(logits[:, :-1, :].reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1))
                    val_loss += loss.item()
                    val_steps += 1
            avg_val = val_loss / val_steps
            history_val.append(avg_val)

        logging.info(f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        # --- SAVE ---
        plot_history(history_train, history_val, exp_dir, epoch)
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': cfg
        }, os.path.join(exp_dir, f"model_ep{epoch+1}.pt"))
        
        # --- GENERATE SAMPLE ---
        if (epoch + 1) % 1 == 0: 
            logging.info("--- Model Thoughts ---")
            model.eval()
            idx = random.randint(0, len(train_ds)-1)
            emb, _ = train_ds[idx]
            fname = train_ds.data[idx][0]
            try:
                cap = model.generate(emb.unsqueeze(0).to(device), tokenizer)
                logging.info(f"[Sample] {fname}: {cap}")
            except Exception as e:
                logging.error(f"Gen failed: {e}")
            logging.info("----------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_idx", type=int, required=True)
    parser.add_argument("--config_file", type=str, default="config/ablation_config.yaml")
    args = parser.parse_args()
    run_training(args.config_idx, args.config_file)
