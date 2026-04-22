import os
import sys
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# CONFIGURATION
# ==========================================
GPU_ID = "2"                # GPU index
MAX_PARALLEL = 2            # How many to run at once
EXPERIMENT_INDICES = list(range(14)) # 0 to 13
LOG_DIR = "./logs_launcher" # Where to save the terminal output of each run

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================
# WORKER FUNCTION
# ==========================================
def run_experiment(idx):
    """
    Runs a single experiment. 
    Redirects its specific output to logs_launcher/exp_X.out
    """
    log_file = os.path.join(LOG_DIR, f"exp_{idx}.out")
    
    print(f"[Launcher] 🚀 Starting Config {idx}... (Log: {log_file})")
    sys.stdout.flush() # Force print to appear immediately in nohup.out

    # Command
    # Unbuffered python (-u) ensures logs appear in real-time
    cmd = [
        "python", "-u", "-m", "training.train_ablation",
        "--config_idx", str(idx)
    ]

    # Environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID

    start_time = time.time()

    # Execute and redirect output to file
    with open(log_file, "w") as f:
        try:
            # We combine stdout and stderr into the file
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
            
            elapsed = (time.time() - start_time) / 60
            print(f"[Launcher] ✅ Config {idx} COMPLETED in {elapsed:.1f} min.")
        
        except subprocess.CalledProcessError:
            print(f"[Launcher] ❌ Config {idx} FAILED. Check {log_file} for details.")
        
    sys.stdout.flush()

# ==========================================
# MAIN
# ==========================================
def main():
    print(f"--- Parallel Launcher Initiated ---")
    print(f"GPU: {GPU_ID} | Max Parallel: {MAX_PARALLEL}")
    print(f"Queue: {EXPERIMENT_INDICES}")
    print(f"Logs: {LOG_DIR}/")
    print("-----------------------------------")
    sys.stdout.flush()

    # Create a pool of workers (threads)
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        # Map the function to the indices
        executor.map(run_experiment, EXPERIMENT_INDICES)

    print("\n[Launcher] 🎉 All experiments finished.")

if __name__ == "__main__":
    main()