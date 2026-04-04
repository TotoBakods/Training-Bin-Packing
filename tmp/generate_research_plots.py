import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ml_utils import MLOptimizer
from optimizer import repair_solution_compact, get_valid_z_positions

# Set styling
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
mpl_params = {
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'grid.alpha': 0.3
}
plt.rcParams.update(mpl_params)

METRICS_PATH = "Documents/04_Machine_Learning/Performance_Metrics/full_run_metrics.json"
OUTPUT_DIR = "Documents/04_Machine_Learning/Performance_Metrics/research_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_box(ax, pos, dims, color='blue', alpha=0.3):
    x, y, z = pos
    l, w, h = dims
    v = np.array([[x, y, z], [x+l, y, z], [x+l, y+w, z], [x, y+w, z],
                  [x, y, z+h], [x+l, y, z+h], [x+l, y+w, z+h], [x, y+w, z+h]])
    faces = [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
             [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
             [v[0], v[3], v[7], v[4]], [v[1], v[2], v[6], v[5]]]
    poly = Poly3DCollection(faces, alpha=alpha, facecolors=color, edgecolors='black', linewidths=0.5)
    ax.add_collection3d(poly)

def generate_ml_plots(data):
    print("Generating ML Research Plots...")
    
    # 1. research_ml_convergence.png
    eo_ga = data['training_results']['model_fit_eo_ga']
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (MSE)', color=color)
    ax1.semilogy(eo_ga['train_history'], color=color, label='Training Loss', alpha=0.8)
    ax1.semilogy(eo_ga['val_history'], '--', color=color, label='Validation Loss', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Fitness (%)', color=color)
    ax2.plot(eo_ga['val_fitness'], color=color, label='Validation Fitness', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title("Neural Architecture Convergence (EO-GA Variant)", fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "research_ml_convergence.png"), dpi=300)
    plt.close()

    # 2. research_ml_error_distribution.png
    mae = eo_ga['per_output_mae'] # X, Y, Z, Theta
    dims = ['X-Coord', 'Y-Coord', 'Z-Coord', 'Rotation']
    plt.figure(figsize=(10, 6))
    sns.barplot(x=dims, y=mae, palette='magma')
    plt.ylabel("Mean Absolute Error (Normalized)")
    plt.title("Regression Error Distribution (Physics-Biased)", fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "research_ml_error_distribution.png"), dpi=300)
    plt.close()

    # 3. research_ml_vu_summary.png
    results_600 = data['inference_results']['600_items.csv']
    models = list(results_600.keys())
    vu_values = [results_600[m]['su_pct'] for m in models]
    labels = ["EO", "EO-GA", "GA", "GA-EO"]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=vu_values, palette='viridis')
    plt.ylabel("Volumetric Utility (%)")
    plt.title("Industrial Volumetric Benchmark (600 SKU Scale)", fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "research_ml_vu_summary.png"), dpi=300)
    plt.close()

def generate_heuristic_plots(data):
    print("Generating Heuristic Research Plots...")
    
    # 1. research_heuristic_pipeline_panels.png
    num_items = 25
    variant = "fit_eo_ga"
    optimizer = MLOptimizer(variant=variant)
    
    dataset_path = "training_data/warehouse_training.csv"
    df = pd.read_csv(dataset_path).head(num_items)
    items_props = np.zeros((num_items, 9))
    for i, row in df.iterrows():
        items_props[i] = [row['item_l'], row['item_w'], row['item_h'], 1, 1, 1, row['weight'], 0, row['fragile']]
    
    wh_dim = (10.0, 10.0, 10.0)
    wh_l, wh_w, wh_h = wh_dim
    features = np.zeros((num_items, 19))
    for i in range(num_items):
        l, w, h = items_props[i, 0], items_props[i, 1], items_props[i, 2]
        features[i] = [l/10.0, w/10.0, h/10.0, items_props[i, 6]/100.0, items_props[i, 8], 1.0, 1.0, wh_l/100.0, wh_w/100.0, wh_h/100.0, (l*w*h)/10.0, 1000.0/1000.0, (l*w*h)/1000.0, (l*w)/10.0, 100.0/100.0, (l*w)/100.0, l/10.0, w/10.0, i/float(num_items)]
        
    with torch.no_grad():
        outputs = optimizer.model(torch.tensor(features, dtype=torch.float32).to(optimizer.device)).cpu().numpy()
    
    raw_coords = np.column_stack([outputs[:, 0] * wh_l, outputs[:, 1] * wh_w, np.maximum(outputs[:, 2] * wh_h, 0), outputs[:, 3] * 6.0])
    # Step 2: Intersection correction
    valid_z = get_valid_z_positions({"length": wh_l, "width": wh_w, "height": wh_h})
    refined_coords = repair_solution_compact(raw_coords.copy(), items_props, (wh_l, wh_w, wh_h, 0, 0), None, valid_z, fast_mode=True)
    
    fig = plt.figure(figsize=(18, 6))
    colors = plt.cm.coolwarm(np.linspace(0, 1, num_items))
    
    stages = [
        ("A: Neural Intent (Stage 1)", raw_coords),
        ("B: Heuristic Projection (Stage 2)", refined_coords),
        ("C: Physical State (Stage 3)", refined_coords) # In this fast mode, refined is the final
    ]
    
    for idx, (title, coords) in enumerate(stages):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        ax.set_xlim(0, wh_l); ax.set_ylim(0, wh_w); ax.set_zlim(0, wh_h)
        ax.set_title(title, fontsize=12, fontweight='bold')
        for i in range(num_items):
            draw_box(ax, coords[i, :3], items_props[i, :3], color=colors[i], alpha=0.5 if idx==0 else 0.7)
        ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "research_heuristic_pipeline_panels.png"), dpi=300)
    plt.close()

    # 2. research_heuristic_repair_latency.png
    scales = [200, 400, 600]
    latencies = {
        "EO": [data['inference_results']['200_items.csv']['model_fit_eo']['repair_ms'], 
               data['inference_results']['400_items.csv']['model_fit_eo']['repair_ms'],
               data['inference_results']['600_items.csv']['model_fit_eo']['repair_ms']],
        "EO-GA": [data['inference_results']['200_items.csv']['model_fit_eo_ga']['repair_ms'],
                  data['inference_results']['400_items.csv']['model_fit_eo_ga']['repair_ms'],
                  data['inference_results']['600_items.csv']['model_fit_eo_ga']['repair_ms']]
    }
    
    plt.figure(figsize=(10, 6))
    for name, vals in latencies.items():
        plt.plot(scales, np.array(vals)/1000.0, marker='o', label=name, linewidth=2)
    plt.xlabel("SKU Count")
    plt.ylabel("Repair Latency (Seconds)")
    plt.title("Pipeline Scaling Efficiency", fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "research_heuristic_repair_latency.png"), dpi=300)
    plt.close()

    # 3. research_heuristic_stability_heatmap.png
    # Simulating a stability heatmap from the refined coords
    grid_res = 20
    heatmap = np.zeros((grid_res, grid_res))
    for i in range(num_items):
        gx = int((refined_coords[i, 0] / wh_l) * grid_res)
        gy = int((refined_coords[i, 1] / wh_w) * grid_res)
        if 0 <= gx < grid_res and 0 <= gy < grid_res:
            heatmap[gy, gx] += 1
            
    plt.figure(figsize=(8, 8))
    sns.heatmap(heatmap, cmap="YlGnBu", square=True, cbar_kws={'label': 'Item Density (Support Contacts)'})
    plt.title("Spatial Stability Heatmap (Warehouse Floor)", fontweight='bold')
    plt.xlabel("Warehouse Width (m)")
    plt.ylabel("Warehouse Length (m)")
    plt.savefig(os.path.join(OUTPUT_DIR, "research_heuristic_stability_heatmap.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    with open(METRICS_PATH, 'r') as f:
        data = json.load(f)
    generate_ml_plots(data)
    generate_heuristic_plots(data)
    print("All research plots generated successfully.")
