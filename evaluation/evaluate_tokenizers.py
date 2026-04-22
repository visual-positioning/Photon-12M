import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import traceback
import json
import torch
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import logging
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from bert_score import score as bert_score

# Import your model and tokenizer builders
from architecture.mobilecap_modern import build_modern_model
from training.experiment_runner import get_or_create_tokenizer, TOKENIZERS_CONFIG 

# ==========================================
# 1. CONFIGURATION & LOGGING SETUP
# ==========================================
EXPERIMENT_ROOT = "./checkpoints_tokenizer_experiments"
RESULTS_DIR = "./evaluation_results"
SUBMISSIONS_DIR = "./submission_files"
RESULTS_CSV = os.path.join(RESULTS_DIR, "all_evaluation_results.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# Set up logging to output to BOTH the file and the terminal console
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s', 
    handlers=[
        logging.FileHandler("evaluation_tokenizers.log", mode='a'),
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)

GENERATION_BATCH_SIZE = 128 
NUM_WORKERS = 4

DATASETS = {
    # "COCO_karpathy_test": {
    #     "annot_path": os.path.expanduser("~/dataset/karapathy_split/dataset_coco.json"), 
    #     "emb_path": os.path.expanduser("~/dataset/MSCOCO2014/embeddings_test.pt")
    # },
    # "Flickr8k": {
    #     "annot_path": os.path.expanduser("~/dataset/karapathy_split/dataset_flickr8k.json"),
    #     "emb_path": os.path.expanduser("~/dataset/flickr8k/embeddings_test.pt")
    # },
    # "Flickr30k": {
    #     "annot_path": os.path.expanduser("~/dataset/karapathy_split/dataset_flickr30k.json"),
    #     "emb_path": os.path.expanduser("~/dataset/flickr30k_images/embeddings_test.pt")
    # },
    "TextCaps": {
        "annot_path": os.path.expanduser("~/dataset/TextCaps/TextCaps_0.1_val.json"), 
        "emb_path": os.path.expanduser("~/dataset/TextCaps/embeddings_val.pt")      
    },
    "NoCaps": {
        "annot_path": os.path.expanduser("~/dataset/nocaps/nocaps_val_captions.json"), 
        "emb_path": os.path.expanduser("~/dataset/nocaps/embeddings_val.pt")          
    }
}

# ==========================================
# 2. DATASET ABSTRACTION
# ==========================================
class EmbeddingDataset(Dataset):
    def __init__(self, emb_dict):
        self.img_ids = list(emb_dict.keys())
        self.embs = list(emb_dict.values())

    def __len__(self): return len(self.img_ids)
    def __getitem__(self, idx): return self.img_ids[idx], self.embs[idx]

# ==========================================
# 3. UTILITIES & EVALUATION LOGIC
# ==========================================
def ensure_standard_coco_format(annot_path):
    """Converts Karpathy or TextCaps JSONs into the strict format pycocoevalcap expects."""
    with open(annot_path, 'r') as f:
        data = json.load(f)

    if 'annotations' in data and 'images' in data:
        return annot_path

    converted_path = annot_path.replace(".json", "_eval_format.json")
    if os.path.exists(converted_path):
        return converted_path 

    logger.info(f"      [i] First time evaluating this dataset: Formatting annotations for pycocoevalcap...")
    standard_data = {"type": "captions", "images": [], "annotations": []}
    ann_id = 0

    # Handle Karpathy Format (COCO / Flickr)
    if 'images' in data and len(data['images']) > 0 and 'sentences' in data['images'][0]:
        for img in data['images']:
            # FIX: Only load the test split!
            if img.get('split') not in ['test', 'val']:
                continue
                
            img_id = img.get('cocoid', img.get('imgid', img.get('filename')))
            standard_data["images"].append({"id": img_id, "file_name": img['filename']})
            for sent in img['sentences']:
                standard_data["annotations"].append({
                    "image_id": img_id,
                    "id": ann_id,
                    "caption": sent['raw']
                })
                ann_id += 1

    # Handle TextCaps Format
    elif 'data' in data:
        for img in data['data']:
            img_id = img['image_id']
            standard_data["images"].append({"id": img_id})
            for cap in img.get('reference_strs', []):
                standard_data["annotations"].append({
                    "image_id": img_id,
                    "id": ann_id,
                    "caption": cap
                })
                ann_id += 1

    with open(converted_path, 'w') as f:
        json.dump(standard_data, f)
    return converted_path

def generate_predictions(model, tokenizer, emb_path, device, desc="Generating"):
    model.eval()
    dataset_embs = torch.load(emb_path, map_location='cpu') 
    dataset = EmbeddingDataset(dataset_embs)
    loader = DataLoader(dataset, batch_size=GENERATION_BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    predictions = []
    with torch.no_grad():
        for img_ids, img_embs in tqdm(loader, desc=desc, leave=False):
            img_embs = img_embs.to(device)
            pred_captions = model.generate(img_embs, tokenizer) 
            
            for i, caption in enumerate(pred_captions):
                img_id = img_ids[i]
                if isinstance(img_id, torch.Tensor): img_id = img_id.item()
                predictions.append({"image_id": img_id, "caption": caption})
            
    return predictions

def calculate_metrics(predictions, annot_path, saved_json_path):
    standard_annot_path = ensure_standard_coco_format(annot_path)
    
    # Suppress pycocoevalcap standard output prints to keep console clean
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(standard_annot_path)
        cocoRes = coco.loadRes(saved_json_path) 
        cocoEval = COCOEvalCap(coco, cocoRes)
        
        # 🛠️ Tell the evaluator to ONLY score the exact images we generated!
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()
    
    metrics = {metric: score for metric, score in cocoEval.eval.items()}
    
    # ==========================================
    # ROBUST BERTScore Calculation
    # ==========================================
    refs, cands = [], []
    
    # 1. Create a dictionary mapping string IDs to predicted captions
    pred_dict = {str(pred["image_id"]): pred["caption"] for pred in predictions}
    
    # 2. Iterate only over the IDs the COCO evaluator successfully loaded
    for img_id in cocoRes.getImgIds():
        str_id = str(img_id)
        
        if str_id in pred_dict:
            # 🛠️ THE FIX: Bypass the buggy pycocotools API and access the internal dictionary directly!
            anns = coco.imgToAnns[img_id]
            
            # 3. Only add to BERTScore if ground truth sentences actually exist
            if len(anns) > 0:
                cands.append(pred_dict[str_id])
                refs.append([ann['caption'] for ann in anns])

    # 4. Safe execution
    if len(cands) == 0:
        logger.warning("      [!] No valid candidate/reference pairs found for BERTScore. Setting to 0.0")
        metrics["BERTScore_F1"] = 0.0
    else:
        P, R, F1 = bert_score(cands, refs, lang="en", rescale_with_baseline=True, verbose=False)
        metrics["BERTScore_F1"] = F1.mean().item()
    
    return metrics
# ==========================================
# 4. MAIN ORCHESTRATION
# ==========================================
def run_evaluation_suite():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    completed_runs = set()
    if os.path.exists(RESULTS_CSV):
        existing_df = pd.read_csv(RESULTS_CSV)
        for _, row in existing_df.iterrows():
            completed_runs.add(f"{row['Dataset']}_{row['Tokenizer']}_{row['Run']}")
        logger.info(f"\n[i] Found {len(completed_runs)} completed runs in existing CSV. Resuming from where left off.")

    dataset_keys = list(DATASETS.keys())
    for d_idx, dataset_name in enumerate(dataset_keys):
        paths = DATASETS[dataset_name]
        logger.info(f"\n" + "="*75)
        logger.info(f"📁 DATASET [{d_idx+1}/{len(dataset_keys)}]: {dataset_name}")
        logger.info(f"="*75)
        
        annot_path = paths["annot_path"]
        emb_path = paths["emb_path"]
        
        tok_keys = list(TOKENIZERS_CONFIG.keys())
        for t_idx, tok_name in enumerate(tok_keys):
            tok_dir = os.path.join(EXPERIMENT_ROOT, tok_name)
            if not os.path.exists(tok_dir):
                continue

            logger.info(f"\n  🧠 TOKENIZER [{t_idx+1}/{len(tok_keys)}]: {tok_name}")
            logger.info(f"  " + "-"*40)
            
            tokenizer = get_or_create_tokenizer(tok_name)
            vocab_size = len(tokenizer) if tok_name != "clip_pretrained" else tokenizer.vocab_size
                
            runs = [d for d in os.listdir(tok_dir) if d.startswith("run_")]
            
            for i, run_folder in enumerate(runs):
                run_idx = int(run_folder.split("_")[1])
                run_signature = f"{dataset_name}_{tok_name}_{run_idx}"
                
                if run_signature in completed_runs:
                    logger.info(f"    ⏭️ Skip: Run {run_idx} ({i+1}/{len(runs)}) - Already evaluated.")
                    continue

                model_path = os.path.join(tok_dir, run_folder, "best_model.pt")
                if not os.path.exists(model_path):
                    continue
                    
                logger.info(f"    🚀 Processing Run {run_idx} ({i+1}/{len(runs)})...")
                
                # Load Model and Generate
                model, _ = build_modern_model(tokenizer=tokenizer, override_cfg={"vocab_size": vocab_size})
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                
                pbar_desc = f"Gen [{tok_name} | Run {run_idx}]"
                predictions = generate_predictions(model, tokenizer, emb_path, device, desc=pbar_desc)
                
                # Save Predictions JSON
                save_dir = os.path.join(SUBMISSIONS_DIR, dataset_name, tok_name)
                os.makedirs(save_dir, exist_ok=True)
                json_save_path = os.path.join(save_dir, f"run_{run_idx}_predictions.json")
                
                with open(json_save_path, 'w') as f:
                    json.dump(predictions, f)
                logger.info(f"      ✅ Generation complete! Saved JSON to: {json_save_path}")

                # Evaluate Metrics
                if annot_path and os.path.exists(annot_path):
                    try:
                        metrics = calculate_metrics(predictions, annot_path, json_save_path)
                        row = {"Dataset": dataset_name, "Tokenizer": tok_name, "Run": run_idx}
                        row.update(metrics)
                        
                        df_incremental = pd.DataFrame([row])
                        if not os.path.isfile(RESULTS_CSV):
                            df_incremental.to_csv(RESULTS_CSV, index=False)
                        else:
                            df_incremental.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
                            
                        logger.info(f"      ✅ Evaluation complete! Metrics appended to {RESULTS_CSV}")
                        completed_runs.add(run_signature) 
                        
                    except Exception as e:
                        logger.error(f"      ❌ Failed to calculate metrics:")
                        logger.error(traceback.format_exc()) # This prints the EXACT line and cause of the crash
                else:
                    logger.info(f"      [i] No annotations provided for {dataset_name}. Skipped local metrics calculation.")

def calculate_statistics():
    if not os.path.exists(RESULTS_CSV):
        logger.warning("No evaluation results found to calculate statistics.")
        return
        
    df = pd.read_csv(RESULTS_CSV)
    if df.empty: return
        
    logger.info("\n" + "="*75)
    logger.info("📊 CALCULATING FINAL STATISTICS")
    logger.info("="*75)
    
    stats_df = df.groupby(["Dataset", "Tokenizer"]).agg(
        {col: ['mean', 'std'] for col in df.columns if col not in ['Dataset', 'Tokenizer', 'Run']}
    ).round(4)
    
    stats_path = os.path.join(RESULTS_DIR, "aggregated_statistics.csv")
    stats_df.to_csv(stats_path)
    logger.info(f"  [+] Saved Mean and Std Dev to: {stats_path}")
    
    baseline_tok = "sentencepiece"
    metrics_to_test = ["CIDEr", "Bleu_4", "BERTScore_F1"]
    p_values_records = []
    
    for dataset in df['Dataset'].unique():
        df_ds = df[df['Dataset'] == dataset]
        baseline_data = df_ds[df_ds['Tokenizer'] == baseline_tok]
        
        if baseline_data.empty: continue
            
        for tok in df['Tokenizer'].unique():
            if tok == baseline_tok: continue
                
            comp_data = df_ds[df_ds['Tokenizer'] == tok]
            if comp_data.empty: continue
            
            row = {"Dataset": dataset, "Comparison": f"{tok} vs {baseline_tok}"}
            for metric in metrics_to_test:
                if metric in comp_data.columns and metric in baseline_data.columns:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        t_stat, p_val = stats.ttest_ind(comp_data[metric], baseline_data[metric], equal_var=False)
                        row[f"{metric}_p_value"] = round(p_val, 5) if not pd.isna(p_val) else 1.0
                        row[f"{metric}_significant"] = row[f"{metric}_p_value"] < 0.05
                
            p_values_records.append(row)
            
    if p_values_records:
        p_df = pd.DataFrame(p_values_records)
        ttest_path = os.path.join(RESULTS_DIR, "statistical_significance_tests.csv")
        p_df.to_csv(ttest_path, index=False)
        logger.info(f"  [+] Saved Statistical Significance tests to: {ttest_path}")
        logger.info("\n🎉 All Processing Complete!")

if __name__ == "__main__":
    run_evaluation_suite()
    calculate_statistics()