import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ml_utils import MLOptimizer

def draw_box(ax, pos, dims, color='blue', alpha=0.3):
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

def generate_combined_raw_plot(num_items=30):
    models_to_check = {
        "Standalone EO": "fit_eo",
        "Standalone GA": "fit_ga",
        "GA-EO Hybrid": "fit_ga_eo",
        "EO-GA Hybrid": "fit_eo_ga"
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
    
    wh_dim = (10.0, 10.0, 10.0)
    wh_l, wh_w, wh_h = wh_dim
    wh_vol = wh_l * wh_w * wh_h
    wh_area = wh_l * wh_w
    
    # Build feature vectors according to ml_utils 20-dim logic
    features = np.zeros((num_items, 20), dtype=np.float32)
    
    item_l_max = max(float(max(item['length'] for item in items)), 1.0)
    item_w_max = max(float(max(item['width'] for item in items)), 1.0)
    item_h_max = max(float(max(item['height'] for item in items)), 1.0)
    weight_max = max(float(max(item.get('weight', 0) for item in items)), 1.0)
    access_freq_max = 1.0
    priority_max = 1.0
    af_prio_max = 1.0
    door_x = 0.0
    door_y = 0.0

    for i, item in enumerate(items):
        l, w, h = item['length'], item['width'], item['height']
        item_vol = l * w * h
        item_area = l * w
        af = item.get('access_freq', 1.0)
        prio = item.get('priority', 1)
        
        features[i] = [
            l / item_l_max,
            w / item_w_max,
            h / item_h_max,
            item.get('weight', 0) / weight_max,
            1.0 if item.get('fragility', 0) else 0.0,
            1.0 if item.get('stackable', 1) else 0.0,
            1.0 if item.get('can_rotate', 1) else 0.0,
            af / access_freq_max,
            prio / priority_max,
            door_x / (wh_l + 1e-6),
            item_vol / max(item_l_max * item_w_max * item_h_max, 1e-6),
            door_y / (wh_w + 1e-6),
            item_vol / (wh_vol + 1e-6),
            item_area / max(item_l_max * item_w_max, 1e-6),
            0.0 if item.get('fragility', 0) else 1.0,
            item_area / (wh_area + 1e-6),
            l / (wh_l + 1e-6),
            w / (wh_w + 1e-6),
            i / float(max(num_items, 1)),
            (af * prio) / af_prio_max,
        ]
        
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Raw MLP Output Predictions\n(Pre-Heuristic Repair)", fontsize=24, fontweight='bold')
    
    colors = plt.cm.plasma(np.linspace(0, 1, num_items))
    
    for idx, (name, variant) in enumerate(models_to_check.items()):
        model_path = f"models/model_{variant}.pth"
        
        ax = fig.add_subplot(2, 2, idx+1, projection='3d')
        ax.set_xlim(0, wh_l)
        ax.set_ylim(0, wh_w)
        ax.set_zlim(0, wh_h)
        ax.set_title(name, fontsize=18, fontweight='bold')
        ax.set_xlabel("Length (m)")
        ax.set_ylabel("Width (m)")
        ax.set_zlabel("Height (m)")
        
        if not os.path.exists(model_path):
            ax.text(wh_l/2, wh_w/2, wh_h/2, "Model Missing", color='red', ha='center', va='center', fontsize=20)
            continue
            
        optimizer = MLOptimizer(variant=variant)
        
        with torch.no_grad():
            outputs = optimizer.model(torch.tensor(features).to(optimizer.device)).cpu().numpy()
            
        pred_x = outputs[:, 0] * wh_l
        pred_y = outputs[:, 1] * wh_w
        pred_z = outputs[:, 2] * wh_h
        
        max_x, max_y, max_z = 0, 0, 0
        min_x, min_y, min_z = wh_l, wh_w, wh_h
        
        for i in range(num_items):
            z_val = np.maximum(pred_z[i], 0)
            pos = (pred_x[i], pred_y[i], z_val)
            dims = (items[i]['length'], items[i]['width'], items[i]['height'])
            draw_box(ax, pos, dims, color=colors[i], alpha=0.5)
            
            # Track bounds for zoom
            min_x = min(min_x, pred_x[i])
            min_y = min(min_y, pred_y[i])
            min_z = min(min_z, z_val)
            
            max_x = max(max_x, pred_x[i] + dims[0])
            max_y = max(max_y, pred_y[i] + dims[1])
            max_z = max(max_z, z_val + dims[2])
            
        # Add a 10% margin
        margin = 0.1
        len_x = max(max_x - min_x, 1.0)
        len_y = max(max_y - min_y, 1.0)
        len_z = max(max_z - min_z, 1.0)
        
        # Re-apply zoomed limits
        ax.set_xlim(max(0, min_x - len_x*margin), min(wh_l, max_x + len_x*margin))
        ax.set_ylim(max(0, min_y - len_y*margin), min(wh_w, max_y + len_y*margin))
        ax.set_zlim(max(0, min_z - len_z*margin), min(wh_h, max_z + len_z*margin))
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = "all_models_raw_3d.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_combined_raw_plot()
