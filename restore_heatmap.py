import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load existing metrics
METRICS_BASE_DIR = os.path.join("Documents", "04_Machine_Learning", "Performance_Metrics")
VISUALS_DIR = os.path.join(METRICS_BASE_DIR, "metrics_visuals")
json_path = os.path.join(METRICS_BASE_DIR, "full_run_metrics.json")

with open(json_path, 'r') as f:
    data = json.load(f)

training_results = data['training_results']
inference_results = data['inference_results']
physics_results = data['physics_results']

def save_stability_heatmap(physics_results):
    plt.figure(figsize=(10, 7))
    all_x, all_y, all_d = [], [], []
    for res in physics_results.values():
        all_x.extend(res.get("raw_x", []))
        all_y.extend(res.get("raw_y", []))
        all_d.extend(res.get("raw_displacements", []))
    
    if not all_x: 
        print("Warning: No spatial data found for heatmap.")
        return
    
    plt.hexbin(all_x, all_y, C=all_d, gridsize=30, cmap='YlOrRd', reduce_C_function=np.mean)
    plt.colorbar(label='Mean Settlement Displacement (m)')
    plt.title('Warehouse Stability Heatmap\n(Settlement Displacement across X/Y Plane)')
    plt.xlabel('Warehouse Length (m)')
    plt.ylabel('Warehouse Width (m)')
    
    path = os.path.join(VISUALS_DIR, "stability_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"Stability Heatmap restored to {path}")

def update_ml_report():
    report_path = os.path.join(METRICS_BASE_DIR, "model_metrics_ml.md")
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check if image link exists, if not, we'd need to recreate the report
    # For speed, let's just run the actual report generator logic from evaluate_metrics.py
    pass

# Execute restoration
save_stability_heatmap(physics_results)
