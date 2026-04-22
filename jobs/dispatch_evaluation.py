import subprocess
import os
import sys
import json
import pandas as pd
import yaml
from concurrent.futures import ThreadPoolExecutor
import time

# ==========================================
# CONFIGURATION
# ==========================================
# Run indices 0 to 13
EXPERIMENT_INDICES = list(range(14))
CONFIG_FILE = "config/ablation_config.yaml"

# GPUs to use
GPUS = [0, 1, 2]

# Log Directory
LOG_DIR = "jobs/logs"

# ==========================================
# UTILS
# ==========================================
def load_exp_name(idx):
    with open(CONFIG_FILE, 'r') as f:
        data = yaml.safe_load(f)
    return data[idx]['experiment_name']

def run_eval_job(gpu_id, config_idx):
    """Runs a single evaluation job on a specific GPU."""
    exp_name = load_exp_name(config_idx)
    log_file = os.path.join(LOG_DIR, f"eval_exp_{config_idx}.log")
    
    print(f"[GPU {gpu_id}] 🚀 Starting Eval: Exp {config_idx} ({exp_name})")
    
    # Command: python -m testing.evaluate_metrics ...
    cmd = [
        sys.executable, "-u", "-m", "testing.evaluate_metrics",
        "--config_idx", str(config_idx),
        "--config_file", CONFIG_FILE
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Run and redirect logs
    with open(log_file, "w") as f:
        try:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
            print(f"[GPU {gpu_id}] ✅ Exp {config_idx} Finished.")
        except subprocess.CalledProcessError:
            print(f"[GPU {gpu_id}] ❌ Exp {config_idx} FAILED. See {log_file}")

def worker_gpu(gpu_id, queue):
    """
    Worker thread that processes a specific queue of experiments for one GPU.
    """
    for idx in queue:
        run_eval_job(gpu_id, idx)

# ==========================================
# MAIN
# ==========================================
def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"🚀 Dispatching Evaluation for {len(EXPERIMENT_INDICES)} experiments across GPUs {GPUS}...")
    print(f"📂 Logs will be saved to: {LOG_DIR}/")
    print("-" * 50)
    
    # 1. Distribute jobs (Round Robin)
    # queue_0 = [0, 3, 6...]
    # queue_1 = [1, 4, 7...]
    # queue_2 = [2, 5, 8...]
    queues = {gpu: [] for gpu in GPUS}
    for i, exp_idx in enumerate(EXPERIMENT_INDICES):
        target_gpu = GPUS[i % len(GPUS)]
        queues[target_gpu].append(exp_idx)
        
    # 2. Start GPU Workers
    with ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
        futures = []
        for gpu in GPUS:
            if queues[gpu]: # Only start if queue is not empty
                print(f"GPU {gpu} Queue: {queues[gpu]}")
                futures.append(executor.submit(worker_gpu, gpu, queues[gpu]))
        
        # Wait for all to finish
        for f in futures:
            f.result()

    print("\n🎉 All evaluation jobs finished.")
    
    # 3. TABULATE RESULTS
    print("\n=============================================")
    print("📊 FINAL RESULTS TABLE")
    print("=============================================")
    
    all_results = []
    
    for idx in EXPERIMENT_INDICES:
        exp_name = load_exp_name(idx)
        metrics_file = f"results/{exp_name}/metrics.json"
        
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                try:
                    m = json.load(f)
                    row = {
                        "ID": idx,
                        "Experiment": exp_name,
                        "CIDEr": round(m.get('CIDEr', 0) * 100, 2),
                        "BLEU-4": round(m.get('Bleu_4', 0) * 100, 2),
                        "METEOR": round(m.get('METEOR', 0) * 100, 2),
                        "SPICE": round(m.get('SPICE', 0) * 100, 2),
                        "BERT_F1": round(m.get('BERTScore_F1', 0) * 100, 2),
                        "CLIP_S": round(m.get('CLIPScore', 0) * 100, 2),
                        "SBERT": round(m.get('SBERT_Sim', 0) * 100, 2)
                    }
                    all_results.append(row)
                except json.JSONDecodeError:
                    all_results.append({"ID": idx, "Experiment": exp_name, "CIDEr": "Error"})
        else:
            all_results.append({"ID": idx, "Experiment": exp_name, "CIDEr": "Missing"})

    if not all_results:
        print("No results found.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns if they exist
    cols = ["ID", "Experiment", "CIDEr", "BLEU-4", "METEOR", "SPICE", "BERT_F1", "CLIP_S", "SBERT"]
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]
    
    # Print nice markdown table
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df) # Fallback if tabulate/markdown not installed
    
    # Save to CSV
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/final_leaderboard.csv", index=False)
    print("\n✅ Results saved to results/final_leaderboard.csv")

if __name__ == "__main__":
    main()