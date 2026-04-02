import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ml_utils import MLOptimizer

def draw_box(ax, pos, dims, color='blue', alpha=0.3, label=None):
    """Draws a 3D box at pos with dims."""
    x, y, z = pos
    l, w, h = dims
    
    # Vertices
    v = np.array([
        [x, y, z], [x+l, y, z], [x+l, y+w, z], [x, y+w, z],
        [x, y, z+h], [x+l, y, z+h], [x+l, y+w, z+h], [x, y+w, z+h]
    ])
    
    # Faces
    faces = [
        [v[0], v[1], v[2], v[3]], # bottom
        [v[4], v[5], v[6], v[7]], # top
        [v[0], v[1], v[5], v[4]], # side 1
        [v[2], v[3], v[7], v[6]], # side 2
        [v[0], v[3], v[7], v[4]], # side 3
        [v[1], v[2], v[6], v[5]]  # side 4
    ]
    
    poly = Poly3DCollection(faces, alpha=alpha, facecolors=color, edgecolors='black', linewidths=0.5)
    ax.add_collection3d(poly)

def generate_3d_raw_plots(num_items=30):
    models_to_check = {
        "Standalone EO": "fit_eo",
        "Standalone GA": "fit_ga",
        "GA-EO Hybrid": "fit_ga_eo",
        "EO-GA Hybrid (Fast)": "fit_eo_ga"
    }
    
    # Load sample metadata
    dataset_path = "training_data/warehouse_training.csv"
    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        return
    
    df = pd.read_csv(dataset_path).head(num_items)
    items = []
    for _, row in df.iterrows():
        items.append({
            'length': row['item_l'],
            'width': row['item_w'],
            'height': row['item_h'],
            'weight': row['weight'],
            'fragility': row['fragile'],
            'stackable': row['stackable']
        })
    
    wh_dim = (10.0, 10.0, 10.0) # Standard 10m warehouse
    
    output_dir = "Documents/04_Machine_Learning/Performance_Metrics/metrics_visuals"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, variant in models_to_check.items():
        model_path = f"models/model_{variant}.pth"
        if not os.path.exists(model_path):
            print(f"Skipping {name}, file {model_path} missing.")
            continue
            
        print(f"Generating 3D plot for {name} (Raw)...")
        optimizer = MLOptimizer(variant=variant)
        
        # Get RAW predictions (pre-repair) using exact 19-feature alignment from ml_utils
        wh_l, wh_w, wh_h = wh_dim
        wh_vol = wh_l * wh_w * wh_h
        wh_area = wh_l * wh_w
        
        features = np.zeros((num_items, 19))
        for i, item in enumerate(items):
            l, w, h = item['length'], item['width'], item['height']
            item_vol = l * w * h
            item_area = l * w
            
            features[i] = [
                l / 10.0, 
                w / 10.0, 
                h / 10.0,
                item.get('weight', 0) / 100.0, 
                1.0 if item.get('fragility', 0) else 0.0,
                1.0 if item.get('stackable', 1) else 0.0,
                1.0, # can_rotate proxy
                wh_l / 100.0,
                wh_w / 100.0,
                wh_h / 100.0,
                item_vol / 10.0,
                wh_vol / 1000.0,
                item_vol / (wh_vol + 1e-6),
                item_area / 10.0,
                wh_area / 100.0,
                item_area / (wh_area + 1e-6),
                l / (wh_l + 1e-6),
                w / (wh_w + 1e-6),
                i / float(num_items)
            ]
            
        with torch.no_grad():
            outputs = optimizer.model(torch.tensor(features, dtype=torch.float32).to(optimizer.device)).cpu().numpy()
            
        # Denormalize
        pred_x = outputs[:, 0] * wh_l
        pred_y = outputs[:, 1] * wh_w
        pred_z = outputs[:, 2] * wh_h
        
        # Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set limits
        ax.set_xlim(0, wh_l)
        ax.set_ylim(0, wh_w)
        ax.set_zlim(0, wh_h)
        ax.set_title(f"Raw Neural Predictions: {name}\n(Prior to Heuristic Logic)", fontsize=14, fontweight='bold')
        
        colors = plt.cm.plasma(np.linspace(0, 1, num_items))
        
        for i in range(num_items):
            pos = (pred_x[i], pred_y[i], np.maximum(pred_z[i], 0))
            dims = (items[i]['length'], items[i]['width'], items[i]['height'])
            draw_box(ax, pos, dims, color=colors[i], alpha=0.4)
            
        ax.set_xlabel("Length (m)")
        ax.set_ylabel("Width (m)")
        ax.set_zlabel("Height (m)")
        
        plot_path = os.path.join(output_dir, f"raw_3d_{name.lower().replace(' ', '_')}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved: {plot_path}")

if __name__ == "__main__":
    generate_3d_raw_plots()
