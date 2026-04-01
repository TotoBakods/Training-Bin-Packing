import matplotlib.pyplot as plt
import numpy as np
import os

# Data derived from full_run_metrics.json (Ablation benchmarks)
labels = ['200 Items', '400 Items', '600 Items']
raw_violations = [100, 100, 100]  # Percentage of items requiring repair
repaired_violations = [0, 0, 0]

mean_displacement = [7.92, 9.17, 10.15] # Meters shifted to reach physical feasibility

OUTPUT_DIR = os.path.join("Documents", "04_Machine_Learning", "Performance_Metrics", "metrics_visuals")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_violation_ablation():
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, raw_violations, width, label='Raw MLP (Pre-Repair)', color='salmon')
    plt.bar(x + width/2, repaired_violations, width, label='Heuristic-Repaired', color='skyblue')
    
    plt.title('Physical Constraint Violations (Ablation)')
    plt.ylabel('Violation Rate (%)')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(OUTPUT_DIR, "violation_ablation.png"))
    plt.close()

def plot_displacement_error():
    plt.figure(figsize=(10, 6))
    plt.plot(labels, mean_displacement, marker='o', linestyle='-', color='purple', label='Mean Adjustment Distance (m)')
    plt.fill_between(labels, 0, mean_displacement, color='purple', alpha=0.1)
    
    plt.title('Regression Accuracy Gap (Displacement Error)')
    plt.ylabel('Displacement Magnitude (Meters)')
    plt.xlabel('Batch Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "displacement_error.png"))
    plt.close()

if __name__ == "__main__":
    plot_violation_ablation()
    plot_displacement_error()
    print("Ablation plots generated.")
