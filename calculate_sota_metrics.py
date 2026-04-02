import os
import torch
import numpy as np
import pandas as pd
from ml_utils import MLOptimizer
from optimizer import repair_solution_compact

def calculate_ssr_and_psr(variant="fit_eo_ga", items_count=600):
    print(f"Calculating SOTA Metrics (SSR/PSR) for {variant}...")
    
    # 1. Load Data
    dataset_path = f"gan/{items_count}_items.csv"
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found!")
        return
    
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
        
    wh_dim = (10.0, 10.0, 10.0)
    
    # 2. Run Inference + Repair
    optimizer = MLOptimizer(variant=variant)
    # Corrected unpacking: returns 4 values
    placed_boxes, fitness, total_latency, metrics = optimizer.optimize(items, {'id': 1, 'length': 10, 'width': 10, 'height': 10})
    
    if not placed_boxes:
        print("No items placed!")
        return

    # 3. Calculate PSR (Placement Success Rate)
    # In our system, all items are "placed", but some might be outside bounds or colliding if repair failed.
    success_count = 0
    for i, box in enumerate(placed_boxes):
        item = items[i]
        l, w, h = item['length'], item['width'], item['height']
        # Check if within 10x10x10
        if (box['x'] >= 0 and box['x'] + l <= 10.0 and
            box['y'] >= 0 and box['y'] + w <= 10.0 and
            box['z'] >= 0 and box['z'] + h <= 10.0):
            success_count += 1
            
    psr = (success_count / len(items)) * 100.0
    
    # 4. Calculate SSR (Support Surface Ratio)
    # For each box, check how much of its bottom area is supported by items below (or floor)
    ssr_values = []
    for i, box in enumerate(placed_boxes):
        item = items[i]
        l, w, h = item['length'], item['width'], item['height']
        
        if box['z'] <= 0.01: # On the floor
            ssr_values.append(100.0)
            continue
            
        # Check support from boxes below
        bottom_z = box['z']
        total_bottom_area = l * w
        supported_area = 0.0
        
        # Simple overlap check for support
        for j, other_box in enumerate(placed_boxes):
            if i == j: continue
            other_item = items[j]
            ol, ow, oh = other_item['length'], other_item['width'], other_item['height']
            
            # If other is directly below this box
            if abs((other_box['z'] + oh) - bottom_z) < 0.05:
                # Calculate XY overlap area
                ix1 = max(box['x'], other_box['x'])
                ix2 = min(box['x'] + l, other_box['x'] + ol)
                iy1 = max(box['y'], other_box['y'])
                iy2 = min(box['y'] + w, other_box['y'] + ow)
                
                if ix2 > ix1 and iy2 > iy1:
                    supported_area += (ix2 - ix1) * (iy2 - iy1)
        
        ssr = (min(supported_area, total_bottom_area) / total_bottom_area) * 100.0
        ssr_values.append(ssr)
        
    avg_ssr = np.mean(ssr_values)
    
    print(f"\n--- SOTA Metrics for {variant} ({items_count} items) ---")
    print(f"Placement Success Rate (PSR): {psr:.2f}%")
    print(f"Average Support Surface Ratio (SSR): {avg_ssr:.2f}%")
    print(f"Min SSR: {np.min(ssr_values):.2f}%")
    print(f"Items with 100% Support: {sum(1 for s in ssr_values if s > 99.9)}")
    print("-------------------------------------------\n")
    
if __name__ == "__main__":
    calculate_ssr_and_psr("fit_eo_ga", 600)
