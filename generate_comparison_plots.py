import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Data extracted from full_run_metrics.json
item_counts = [200, 400, 600]
our_bbox_eff = [36.3, 39.9, 45.4] # From EO model which performed best in packing
sota_benchmark = [75.0, 78.0, 82.0] # Representing literature average (Zhao et al., Zhao 2021)

OUTPUT_DIR = os.path.join("Documents", "04_Machine_Learning", "Performance_Metrics", "metrics_visuals")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_utilization_gap():
    plt.figure(figsize=(10, 6))
    plt.plot(item_counts, sota_benchmark, 'g--', label='Zhao et al. (2021) Benchmark (75-82%)', marker='o')
    plt.plot(item_counts, our_bbox_eff, 'b-', label='Our MLP-Heuristic Wrapper', marker='s')
    
    plt.fill_between(item_counts, our_bbox_eff, sota_benchmark, color='red', alpha=0.1, label='Optimization Gap')
    
    plt.title('Volume Utilization (Bounding Box Efficiency) vs. SOTA')
    plt.xlabel('Number of Items (SKUs)')
    plt.ylabel('Volume Utilization (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "research_utilization_gap.png"))
    plt.close()

def plot_inference_scalability():
    # Extracted from inference_ms vs total_ms
    our_infer = [2.5, 5.4, 7.7]
    repair_time = [4340, 13700, 28200]
    
    plt.figure(figsize=(10, 6))
    plt.bar(np.array(item_counts)-20, our_infer, width=40, label='ML Inference Time (ms)', color='blue')
    plt.title('ML Inference Scalability (ms)')
    plt.xlabel('Item Count')
    plt.ylabel('Inference Latency (ms)')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "inference_scalability.png"))
    plt.close()

if __name__ == "__main__":
    plot_utilization_gap()
    plot_inference_scalability()
    print("New comparison plots generated.")
