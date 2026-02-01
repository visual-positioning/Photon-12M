"""
Image Captioning Evaluation Framework
"""

import argparse
import os
import sys
from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Internal modules (Assumed to be in the same workspace)
from common.registry import scan_directory, dynamic_import
from common.io_utils import append_to_json

def main():
    parser = argparse.ArgumentParser(description="Evaluation Framework")
    
    # --- Experiment Configuration ---
    parser.add_argument('--dataset', nargs='+', default=['all'], help='List of datasets (e.g., coco2017 flickr30k)')
    parser.add_argument('--method', nargs='+', default=['all'], help='List of methods (e.g., blip oscar)')
    parser.add_argument('--metric', nargs='+', default=['all'], help='List of metrics (e.g., bleu cider)')
    
    # --- Paths & Metadata ---
    parser.add_argument('--data_root', type=str, required=True, help='Base directory for dataset images')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path for custom ablation checkpoints')
    parser.add_argument('--output', type=str, default='output.json', help='Path to save JSON results')
    
    # --- Reproducibility ---
    parser.add_argument('--experiment_id', type=str, default=None, help='Unique ID (auto-generated if None)')
    parser.add_argument('--remark', type=str, default="", help='Notes regarding this specific run')
    parser.add_argument('--batch_size', type=int, default=8, help='Reduce if memory errors occur')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--verbose', action='store_true', help='If set, prints results to console. Default is silent.')

    args = parser.parse_args()

    # 1. Setup Metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exp_id = args.experiment_id if args.experiment_id else f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 2. Resolve Modules (Combinatorial Logic)
    datasets = scan_directory('materials') if 'all' in args.dataset else args.dataset
    methods = scan_directory('methods') if 'all' in args.method else args.method
    metrics = scan_directory('metrics') if 'all' in args.metric else args.metric

    # Special handling for 'custom' method which requires manual args
    if 'custom' in args.method and 'custom' not in methods:
        methods.append('custom')

    print(f"[{timestamp}] Starting Experiment: {exp_id}")
    print(f"Plan: {len(datasets)} Datasets × {len(methods)} Methods × {len(metrics)} Metrics")

    # 3. Execution Loop
    for d_name in datasets:
        # --- Load Dataset ---
        DatasetClass = dynamic_import(f"materials.{d_name}", "CaptionDataset")
        if not DatasetClass: continue
        
        try:
            d_path = os.path.join(args.data_root, d_name.upper())
            dataset = DatasetClass(root=d_path, split='val')
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        except Exception as e:
            print(f"[Error] Failed to load dataset {d_name}: {e}"); continue

        for m_name in methods:
            # --- Load Method ---
            MethodClass = dynamic_import(f"methods.{m_name}", "CaptionModel")
            if not MethodClass: continue

            try:
                # Inject checkpoint if it's the custom ablation method
                if m_name == 'custom':
                    if not args.checkpoint: 
                        print("[Error] 'custom' method requires --checkpoint. Skipping."); continue
                    model = MethodClass(checkpoint_path=args.checkpoint, device=args.device)
                else:
                    model = MethodClass(device=args.device)
            except Exception as e:
                print(f"[Error] Failed to init method {m_name}: {e}"); continue

            # --- Warmup: Verify model is fully loaded with a test image ---
            try:
                from PIL import Image
                dummy_img = Image.new('RGB', (384, 384), color='white')
                print(f"   [{m_name.upper()}] Running warmup test...")
                warmup_result = model.generate(dummy_img) if hasattr(model, 'generate') else model.generate_batch([dummy_img])[0]
                print(f"   [{m_name.upper()}] Warmup complete. Model ready. Test output: '{warmup_result[:50]}...'")
                del dummy_img  # Free memory
            except Exception as e:
                print(f"[Error] Model warmup failed for {m_name}: {e}")
                continue

            # --- Inference Phase ---
            results = {}       # Format: {image_id: [caption]}
            ground_truths = {} # Format: {image_id: [ref1, ref2]}
            
            # Progress bar is the only console output usually allowed in research scripts
            pbar = tqdm(dataloader, desc=f"{m_name} on {d_name}", unit="batch")
            
            for batch in pbar:
                imgs = batch['image'] # Expecting List[PIL.Image] or Tensor
                ids = batch['image_id']
                refs = batch['captions']
                
                # Generate
                if hasattr(model, 'generate_batch'):
                    preds = model.generate_batch(imgs)
                else:
                    preds = [model.generate(img) for img in imgs]
                
                # Store
                for i, img_id in enumerate(ids):
                    results[img_id] = [preds[i]]
                    ground_truths[img_id] = refs[i]

            # --- Metric Calculation Phase ---
            computed_scores = {}
            for met_name in metrics:
                MetricClass = dynamic_import(f"metrics.{met_name}", "Metric")
                if not MetricClass: continue
                
                try:
                    metric = MetricClass()
                    # Compute returns dict: {'BLEU-1': 80.4, ...}
                    scores = metric.compute(results, ground_truths)
                    computed_scores.update(scores)
                except Exception as e:
                    print(f"[Warning] Metric {met_name} failed: {e}")

            # --- Save Results (Persistent Storage) ---
            entry = {
                "experiment_id": exp_id,
                "timestamp": timestamp,
                "remark": args.remark,
                "configuration": {
                    "dataset": d_name,
                    "method": m_name,
                    "checkpoint": args.checkpoint if m_name == 'custom' else "default",
                    "metrics_list": metrics
                },
                "results": computed_scores
            }
            
            append_to_json(args.output, entry)

            # Optional Verbose Output (Only if requested)
            if args.verbose:
                print(f"Results for {m_name} on {d_name}: {computed_scores}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluation Complete. Data saved to {args.output}")

if __name__ == "__main__":
    main()