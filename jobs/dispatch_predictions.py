import subprocess
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# CONFIGURATION
# ==========================================
# GPUs available
GPUS = [0, 1, 2]

# Max concurrent jobs PER GPU
JOBS_PER_GPU = 4

# Experiments to run (0 to 39)
EXPERIMENT_INDICES = list(range(14))

# ==========================================
# DISPATCHER LOGIC
# ==========================================

def run_job(gpu_id, config_idx):
    """Runs a single prediction job on a specific GPU."""
    log_dir = "logs_gen"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"gen_exp_{config_idx}.log")
    
    print(f"[GPU {gpu_id}] Starting Exp {config_idx}...")
    
    cmd = [
        sys.executable, "-m", "testing.generate_predictions",
        "--config_idx", str(config_idx)
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    with open(log_file, "w") as f:
        try:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
            print(f"[GPU {gpu_id}] ✅ Exp {config_idx} Done.")
        except subprocess.CalledProcessError:
            print(f"[GPU {gpu_id}] ❌ Exp {config_idx} FAILED.")

def worker_gpu(gpu_id, queue):
    """
    A worker thread dedicated to a specific GPU.
    It pulls jobs from its assigned queue and runs them 'JOBS_PER_GPU' at a time.
    """
    with ThreadPoolExecutor(max_workers=JOBS_PER_GPU) as executor:
        # Submit all jobs assigned to this GPU to the thread pool
        futures = [executor.submit(run_job, gpu_id, idx) for idx in queue]
        
        # Wait for all to finish
        for f in futures:
            f.result()

def main():
    print(f"🚀 Dispatching {len(EXPERIMENT_INDICES)} jobs across GPUs {GPUS}")
    print(f"🔥 Max concurrency: {JOBS_PER_GPU} jobs per GPU")
    print("-" * 40)
    
    # 1. Distribute jobs Round-Robin style
    # queue_0 = [0, 3, 6...]
    # queue_1 = [1, 4, 7...]
    # queue_2 = [2, 5, 8...]
    queues = {gpu: [] for gpu in GPUS}
    for i, exp_idx in enumerate(EXPERIMENT_INDICES):
        target_gpu = GPUS[i % len(GPUS)]
        queues[target_gpu].append(exp_idx)
        
    # 2. Start GPU Managers
    with ThreadPoolExecutor(max_workers=len(GPUS)) as main_executor:
        futures = []
        for gpu in GPUS:
            print(f"GPU {gpu} Queue: {queues[gpu]}")
            futures.append(main_executor.submit(worker_gpu, gpu, queues[gpu]))
            
        # Wait for all GPUs to finish
        for f in futures:
            f.result()
            
    print("\n🎉 All prediction jobs completed.")

if __name__ == "__main__":
    main()