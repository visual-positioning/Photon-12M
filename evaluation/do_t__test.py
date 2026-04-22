import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
print(1)
# ==========================================
# 1. CONFIGURATION
# ==========================================
# Update this to match the exact name of the CSV your main script is writing to
RESULTS_CSV = "./evaluation_results/all_evaluation_results.csv" 
OUT_DIR = "./evaluation_results"
BASELINE_TOKENIZER = "coco_bpe"  # Setting the baseline for T-tests

def run_standalone_statistics():
    print(f"Loading data from: {RESULTS_CSV}")
    if not os.path.exists(RESULTS_CSV):
        print(f"❌ Error: Could not find {RESULTS_CSV}")
        return

    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        print("❌ Error: CSV is empty.")
        return

    # Automatically detect all metric columns (ignoring the metadata columns)
    metric_cols = [col for col in df.columns if col not in ['Dataset', 'Tokenizer', 'Run']]
    
    print("\n" + "="*60)
    print("📊 1. CALCULATING MEAN & STD DEV (30 RUNS)")
    print("="*60)
    
    # Calculate Mean and Std for every metric, grouped by Dataset and Tokenizer
    stats_df = df.groupby(["Dataset", "Tokenizer"])[metric_cols].agg(['mean', 'std']).round(4)
    
    # Flatten the multi-level columns for a cleaner CSV output
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
    
    mean_std_path = os.path.join(OUT_DIR, "standalone_aggregated_statistics.csv")
    stats_df.to_csv(mean_std_path)
    print(f"✅ Saved Mean & Std Dev for all metrics to: {mean_std_path}")

    print("\n" + "="*60)
    print(f"🧪 2. CALCULATING T-TESTS (BASELINE: {BASELINE_TOKENIZER})")
    print("="*60)
    
    p_values_records = []
    
    for dataset in df['Dataset'].unique():
        df_ds = df[df['Dataset'] == dataset]
        baseline_data = df_ds[df_ds['Tokenizer'] == BASELINE_TOKENIZER]
        
        if baseline_data.empty:
            print(f"  [!] Warning: Baseline '{BASELINE_TOKENIZER}' missing for {dataset}. Skipping T-tests for this dataset.")
            continue
            
        for tok in df['Tokenizer'].unique():
            if tok == BASELINE_TOKENIZER:
                continue
                
            comp_data = df_ds[df_ds['Tokenizer'] == tok]
            if comp_data.empty: 
                continue
            
            row = {"Dataset": dataset, "Comparison": f"{tok} vs {BASELINE_TOKENIZER}"}
            
            # Dynamically run T-tests for ALL metrics found in the CSV
            for metric in metric_cols:
                if metric in comp_data.columns and metric in baseline_data.columns:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Welch's T-test (equal_var=False) is best practice here
                        t_stat, p_val = stats.ttest_ind(comp_data[metric], baseline_data[metric], equal_var=False)
                        
                        # Store T-Value
                        row[f"{metric}_t_value"] = round(t_stat, 5) if not pd.isna(t_stat) else 0.0
                        
                        # Store P-Value
                        row[f"{metric}_p_value"] = round(p_val, 5) if not pd.isna(p_val) else 1.0
                        
                        # Store Significance flag
                        row[f"{metric}_significant (p<0.05)"] = row[f"{metric}_p_value"] < 0.05
                        
            p_values_records.append(row)
            
    if p_values_records:
        p_df = pd.DataFrame(p_values_records)
        ttest_path = os.path.join(OUT_DIR, "standalone_significance_tests.csv")
        p_df.to_csv(ttest_path, index=False)
        print(f"✅ Saved T-Test results to: {ttest_path}")
    else:
        print("  [!] No T-test comparisons could be made.")

    print("\n🎉 Analysis Complete!")

if __name__ == "__main__":
    run_standalone_statistics()