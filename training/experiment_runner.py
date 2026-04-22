import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import random
import logging
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer

# Suppress HuggingFace Tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import your model components
from architecture.mobilecap_modern import build_modern_model

# ==========================================
# 1. GLOBAL CONFIGURATION & PATHS
# ==========================================
BASE_DIR = os.path.expanduser("~/dataset/MSCOCO2014")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings_s1_train_restval") 
KARPATHY_JSON = os.path.join(os.path.expanduser("~"), "dataset", "karapathy_split", "dataset_coco.json")
ANNOT_TRAIN_PATH = os.path.join(BASE_DIR, "annotations/captions_train2014.json")
ANNOT_VAL_PATH   = os.path.join(BASE_DIR, "annotations/captions_val2014.json")

EXPERIMENT_ROOT = "./checkpoints_tokenizer_experiments"
NUM_RUNS_PER_TOKENIZER = 30
MAX_EPOCHS = 20  
LR = 3e-4
WEIGHT_DECAY = 1e-2
VOCAB_SIZE = 8000

os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
TOKENIZER_DIR = "./tokenizers"
os.makedirs(TOKENIZER_DIR, exist_ok=True)

TOKENIZERS_CONFIG = {
    "common_crawl_bpe": os.path.join(TOKENIZER_DIR, "common_crawl_bpe_8000.json"),
    "clip_pretrained": "openai/clip-vit-base-patch32",
    "sentencepiece": os.path.join(TOKENIZER_DIR, "coco_sentencepiece_8000.json"), 
    "coco_bpe": os.path.join(TOKENIZER_DIR, "coco_bpe_8000.json") 
}

HEAVY_TOKENIZERS = ["clip_pretrained", "common_crawl_bpe"]
EFFECTIVE_BATCH_SIZE = 512

# ==========================================
# 2. LOGGING UTILITIES
# ==========================================
def setup_logger(run_dir):
    logger = logging.getLogger(run_dir)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
        
    log_file = os.path.join(run_dir, "training_process.log")
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def plot_loss_curve(train_losses, val_losses, run_dir):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'loss_curve.png'))
    plt.close()

def generate_and_log_samples(model, dataset, tokenizer, epoch, run_dir, split_name, device, num_samples=3):
    model.eval()
    sample_file = os.path.join(run_dir, f"{split_name}_samples_ep{epoch}.txt")
    
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(f"--- Epoch {epoch} | {split_name.upper()} Set Samples ---\n\n")
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        for idx in indices:
            emb_idx, gt_caption = dataset.final_index[idx]
            img_emb = dataset.embeddings_tensor[emb_idx].unsqueeze(0).to(device)
            try:
                pred_caption = model.generate(img_emb, tokenizer)
            except Exception as e:
                pred_caption = f"[Generation Error: {e}]"
            
            f.write(f"[GT]:   {gt_caption}\n")
            f.write(f"[PRED]: {pred_caption}\n")
            f.write("-" * 50 + "\n")

# ==========================================
# 3. TOKENIZER LOGIC
# ==========================================
def load_train_captions():
    with open(KARPATHY_JSON, "r") as f:
        data = json.load(f)
    return [sent["raw"] for img in data["images"] if img["split"] == "train" for sent in img["sentences"]]

def train_custom_bpe(save_path):
    captions = load_train_captions()
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, min_frequency=2,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(captions, trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))],
    )
    tokenizer.save(save_path)

def train_custom_sentencepiece(save_path):
    captions = load_train_captions()
    tokenizer = SentencePieceUnigramTokenizer()
    tokenizer.train_from_iterator(
        captions, vocab_size=VOCAB_SIZE, show_progress=True,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], unk_token="[UNK]" 
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))],
    )
    tokenizer.save(save_path)

def get_or_create_tokenizer(tok_type):
    path = TOKENIZERS_CONFIG[tok_type]
    if tok_type == "coco_bpe":
        if not os.path.exists(path): train_custom_bpe(path)
        return PreTrainedTokenizerFast(tokenizer_file=path)
    elif tok_type == "sentencepiece":
        if not os.path.exists(path): train_custom_sentencepiece(path)
        return PreTrainedTokenizerFast(tokenizer_file=path)
    elif tok_type in ["clip_pretrained", "common_crawl_bpe"]:
        hf_id = "gpt2" if tok_type == "common_crawl_bpe" else path
        return AutoTokenizer.from_pretrained(hf_id)

# ==========================================
# 4. DATASET DEFINITION
# ==========================================
class KarpathyCocoDataset(Dataset):
    def __init__(self, pt_file_path, annot_paths, tokenizer, max_len=32):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        if not os.path.exists(pt_file_path): raise FileNotFoundError(f"Missing {pt_file_path}")
        pt_data = torch.load(pt_file_path, map_location="cpu", weights_only=False)
        
        self.image_identifiers = []
        embeddings_list = []

        first_key = next(iter(pt_data.keys()))
        if isinstance(first_key, (str, np.str_)) and "COCO" in str(first_key):
            for k, v in pt_data.items():
                self.image_identifiers.append(str(k))
                if isinstance(v, np.ndarray): v = torch.from_numpy(v)
                embeddings_list.append(v)
        else:
            keys = pt_data.keys()
            emb_key = 'embeddings' if 'embeddings' in keys else 'features'
            path_key = 'paths' if 'paths' in keys else ('ids' if 'ids' in keys else None)
            self.image_identifiers = [str(p) for p in pt_data[path_key]]
            embeddings_list = pt_data[emb_key]

        if isinstance(embeddings_list, list): self.embeddings_tensor = torch.stack(embeddings_list).float()
        else: self.embeddings_tensor = embeddings_list.float()

        self.id_to_captions = {}
        for ann_path in annot_paths:
            with open(ann_path, 'r') as f: coco = json.load(f)
            id_to_filename = {img['id']: img['file_name'] for img in coco['images']}
            for ann in coco['annotations']:
                img_id = ann['image_id']
                caption = ann['caption']
                if img_id not in self.id_to_captions: self.id_to_captions[img_id] = []
                self.id_to_captions[img_id].append(caption)
                if img_id in id_to_filename:
                    fname = id_to_filename[img_id]
                    if fname not in self.id_to_captions: self.id_to_captions[fname] = []
                    self.id_to_captions[fname].append(caption)

        self.final_index = []
        for idx, identifier in enumerate(self.image_identifiers):
            key = str(identifier)
            if "/" in key: key = os.path.basename(key)
            captions = self.id_to_captions.get(key, [])
            if not captions:
                try:
                    parsed_id = int(key.split('_')[-1].split('.')[0])
                    captions = self.id_to_captions.get(parsed_id, [])
                except: pass
            for cap in captions: self.final_index.append((idx, cap))

    def __len__(self): return len(self.final_index)
    def __getitem__(self, idx):
        emb_idx, caption = self.final_index[idx]
        img_emb = self.embeddings_tensor[emb_idx]
        tokens = self.tokenizer.encode(caption)
        if len(tokens) > self.max_len: tokens = tokens[:self.max_len]
        pad_needed = self.max_len - len(tokens)
        input_ids = tokens + [self.tokenizer.pad_token_id] * pad_needed
        return img_emb, torch.tensor(input_ids, dtype=torch.long)

# ==========================================
# 5. TRAINING UTILITIES
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    steps = 0
    with torch.no_grad():
        for img_emb, input_ids in loader:
            img_emb, input_ids = img_emb.to(device), input_ids.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(img_emb, input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
            steps += 1
    return total_loss / steps if steps > 0 else 0.0

# ==========================================
# 6. TRAINING WORKER (SINGLE GPU)
# ==========================================
def run_training_worker(tok_name, run_idx, seed, phys_batch, accum_steps):
    """Sequential training function running on a single GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run_dir = os.path.join(EXPERIMENT_ROOT, tok_name, f"run_{run_idx}")
    os.makedirs(run_dir, exist_ok=True)
    
    logger = setup_logger(run_dir)
    logger.info(f"{'='*50}")
    logger.info(f"🚀 RUN {run_idx}/{NUM_RUNS_PER_TOKENIZER} | Tokenizer: {tok_name} | Seed: {seed}")
    logger.info(f"⚙️ Execution: Single GPU | Phys_Batch: {phys_batch} | Accum: {accum_steps}")
    
    loss_log_path = os.path.join(run_dir, "loss.log")
    with open(loss_log_path, "w") as f: f.write("epoch,train_loss,val_loss\n")

    set_seed(seed)
    
    tokenizer = get_or_create_tokenizer(tok_name)
    if tok_name != "clip_pretrained":
        tokenizer.add_special_tokens({
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "mask_token": "[MASK]"
        })
    else:
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    
    model, _ = build_modern_model(tokenizer=tokenizer, override_cfg={"vocab_size": len(tokenizer)}) 
    model.to(device)

    annot_files = [ANNOT_TRAIN_PATH, ANNOT_VAL_PATH]
    train_ds = KarpathyCocoDataset(os.path.join(EMBEDDINGS_DIR, "embeddings_train.pt"), annot_files, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=phys_batch, shuffle=True, num_workers=4, pin_memory=True)

    val_ds = KarpathyCocoDataset(os.path.join(EMBEDDINGS_DIR, "embeddings_val.pt"), annot_files, tokenizer)
    val_loader = DataLoader(val_ds, batch_size=phys_batch, shuffle=False, num_workers=4)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda') 
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    best_val_loss = float('inf')
    best_epoch = 0
    history_train_loss, history_val_loss = [], []

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss, steps = 0, 0
        optimizer.zero_grad() 
        
        is_background = not sys.stdout.isatty()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1} [Run {run_idx}]", disable=is_background)
        
        for i, (img_emb, input_ids) in enumerate(pbar):
            img_emb, input_ids = img_emb.to(device), input_ids.to(device)
            
            with torch.amp.autocast('cuda'):
                logits = model(img_emb, input_ids)
                loss = loss_fn(logits[:, :-1, :].contiguous().view(-1, logits.size(-1)), input_ids[:, 1:].contiguous().view(-1))
                loss = loss / accum_steps 
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += (loss.item() * accum_steps) 
            steps += 1

        avg_train_loss = train_loss / steps
        history_train_loss.append(avg_train_loss)

        avg_val_loss = evaluate(model, val_loader, loss_fn, device)
        history_val_loss.append(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}/{MAX_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        with open(loss_log_path, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")
            
        generate_and_log_samples(model, train_ds, tokenizer, epoch+1, run_dir, "train", device)
        generate_and_log_samples(model, val_ds, tokenizer, epoch+1, run_dir, "val", device)

        if avg_val_loss < best_val_loss:
            logger.info(f"✨ New best val loss: {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            best_epoch = epoch
            
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            
            with open(os.path.join(run_dir, "run_info.json"), "w") as f:
                json.dump({"tokenizer": tok_name, "seed": seed, "best_epoch": best_epoch + 1, "best_val_loss": best_val_loss}, f, indent=4)
        else:
            logger.info(f"🛑 Val loss increased ({avg_val_loss:.4f} > {best_val_loss:.4f}). Stopping early.")
            break

    plot_loss_curve(history_train_loss, history_val_loss, run_dir)
    logger.info(f"✅ Run {run_idx} complete. Best Val: {best_val_loss:.4f} @ Ep {best_epoch+1}")

# ==========================================
# 7. MAIN ORCHESTRATOR
# ==========================================
def main():
    print(f"Starting Single-GPU Sequential Tokenizer Experiment Suite.")
    print(f"Total Target Effective Batch Size: {EFFECTIVE_BATCH_SIZE}")
    
    print("Pre-verifying custom tokenizers...")
    for tok_name in ["coco_bpe", "sentencepiece"]:
        get_or_create_tokenizer(tok_name)

    base_seed = 42
    runs = [(i+1, base_seed + i * 100) for i in range(0, NUM_RUNS_PER_TOKENIZER )]

    for tok_name in TOKENIZERS_CONFIG.keys():
        print(f"\n{'#'*60}\n### PROCESS STRATEGY: {tok_name.upper()} ###\n{'#'*60}")
        
        if tok_name in HEAVY_TOKENIZERS:
            # STRATEGY: 32 physical batch size, accumulate to hit 512
            phys_batch = 256
            accum_steps = max(1, EFFECTIVE_BATCH_SIZE // phys_batch) # Will be 16
            print(f"Type: HEAVY. Running sequentially on Single GPU. Phys Batch: {phys_batch}, Accum Steps: {accum_steps}")
        else:
            # STRATEGY: Light tokenizers can still utilize 512 physical memory sequentially
            phys_batch = 512
            accum_steps = max(1, EFFECTIVE_BATCH_SIZE // phys_batch) # Will be 1
            print(f"Type: LIGHT. Running sequentially on Single GPU. Phys Batch: {phys_batch}, Accum Steps: {accum_steps}")
            
        for run_idx, seed in runs:
            run_training_worker(tok_name, run_idx, seed, phys_batch, accum_steps)

if __name__ == "__main__":
    main()