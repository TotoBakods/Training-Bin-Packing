import os
import torch
import numpy as np
import pandas as pd
from ml_utils import MLOptimizer
from optimizer import repair_solution_compact

def calculate_all_sota_metrics():
    variants = ["fit_regression", "fit_ga", "fit_eo", "fit_eo_ga"]
    scales = [200, 400, 600]
    
    all_results = []
    
    print("--- COMPREHENSIVE SOTA METRICS EVALUATION ---")
    
    # Ensure gan/ directory exists for the results file
    os.makedirs("gan", exist_ok=True)
    
    for items_count in scales:
        for variant in variants:
            print(f"Evaluating {variant} at scale {items_count}...")
            
            try:
                # 1. Load Data
                dataset_path = f"gan/{items_count}_items.csv"
                if not os.path.exists(dataset_path):
                    # Try alternate path
                    dataset_path = f"datasets/sample_{items_count}.csv"
                    if not os.path.exists(dataset_path):
                        print(f"Dataset {dataset_path} not found, skipping scale {items_count} for {variant}.")
                        continue
                
                df = pd.read_csv(dataset_path)
                items = []
                for _, row in df.iterrows():
                    items.append({
                        'id': row.get('id', 'N/A'),
                        'length': row['length'],
                        'width': row['width'],
                        'height': row['height'],
                        'weight': row['weight'],
                        'stackable': row['stackable'],
                        'fragility': row['fragility'],
                        'can_rotate': row.get('can_rotate', 1),
                        'access_freq': row.get('access_freq', 0.5),
                        'priority': row.get('priority', 1),
                        'category': row.get('category', 'general')
                    })
                    
                wh_dim = {'id': 1, 'length': 10, 'width': 10, 'height': 10}
                
                # 2. Run Inference + Repair (Metrics are calculated internally in MLOptimizer)
                optimizer = MLOptimizer(variant=variant)
                placed_boxes, fitness, total_latency, metrics = optimizer.optimize(items, wh_dim)
                
                if not metrics:
                    print(f"No metrics returned for {variant}!")
                    continue

                all_results.append({
                    "Variant": variant,
                    "Scale": items_count,
                    "PSR (%)": round(metrics.get('psr_pct', 0), 2),
                    "SSR (%)": round(metrics.get('ssr_pct', 0), 2),
                    "VU (%)": round(metrics.get('volumetric_efficiency', 0) * 100, 2),
                    "Latency (ms)": round(metrics.get('total_latency_ms', 0), 2)
                })
                
            except Exception as e:
                print(f"Error evaluating {variant} at scale {items_count}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv("gan/research_comparison_metrics.csv", index=False)
        print("\n--- RESULTS SUMMARY ---")
        print(results_df)
        print("\nResults saved to gan/research_comparison_metrics.csv")
    else:
        print("No results collected.")

if __name__ == "__main__":
    calculate_all_sota_metrics()
