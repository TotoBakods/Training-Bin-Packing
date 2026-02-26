import csv
import random
import uuid
import numpy as np
import time
import os
from optimizer import GeneticAlgorithm, ExtremalOptimization, HybridOptimizer, create_random_solution_array
from database import get_warehouse_config

# Configuration
OUTPUT_DIR = "training_data"
SAMPLES_PER_ALGO = 2  # Reduced for speed
ITEMS_PER_SAMPLE = 10  # Reduced for speed
WAREHOUSE_ID = 1

def generate_random_items(count, warehouse_dims):
    items = []
    wh_len, wh_wid, wh_hgt = warehouse_dims
    
    categories = ['Electronics', 'Furniture', 'Clothing', 'Books', 'Toys', 'Auto Parts']
    
    for i in range(count):
        cat = random.choice(categories)
        fragile = random.choice([True, False]) if cat in ['Electronics', 'Toys'] else False
        stackable = not fragile and random.random() > 0.3
        
        # Dimensions (smaller typically fits better for tetris)
        l = round(random.uniform(0.5, 2.0), 2)
        w = round(random.uniform(0.5, 2.0), 2)
        h = round(random.uniform(0.2, 1.0), 2)
        
        item = {
            'id': str(uuid.uuid4()),
            'name': f"GenItem_{i}",
            'length': l, 'width': w, 'height': h,
            'weight': round(random.uniform(2.0, 50.0), 1),
            'category': cat,
            'priority': random.choice([1, 2, 3]),
            'fragility': fragile,
            'stackable': stackable,
            'access_freq': round(random.random(), 3),
            'can_rotate': not fragile,
            # Init pos
            'x': 0, 'y': 0, 'z': 0, 'rotation': 0
        }
        items.append(item)
    return items

def run_generation():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Mock warehouse config if DB not available or just hardcode for speed
    warehouse = {
        'id': 1,
        'length': 20, 'width': 20, 'height': 10,
        'door_x': 0, 'door_y': 0,
        'layer_heights': [] 
    }
    
    weights = {'space': 0.6, 'accessibility': 0.1, 'stability': 0.3} # High priority on Space & Stability for Tetris
    
    algorithms = [
        ('fit_eo', ExtremalOptimization(iterations=10)),
        ('fit_ga', GeneticAlgorithm(population_size=10, generations=5)),
        ('fit_eo_ga', HybridOptimizer(ga_generations=5, eo_iterations=10)),
        ('fit_ga_eo', HybridOptimizer(ga_generations=5, eo_iterations=10))
    ]

    print(f"Starting data generation: {SAMPLES_PER_ALGO} samples per algorithm...")
    
    for algo_name, optimizer in algorithms:
        csv_path = os.path.join(OUTPUT_DIR, f"{algo_name}.csv")
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Header: Item Features + Target (x,y,z,rot) + Warehouse info
            # We record ONE row per ITEM
            if not file_exists:
                header = [
                    'item_l', 'item_w', 'item_h', 'weight', 'fragile', 'stackable', 'can_rotate',
                    'wh_l', 'wh_w', 'wh_h',
                    'target_x', 'target_y', 'target_z', 'target_rot'
                ]
                writer.writerow(header)
            
            samples_collected = 0
            while samples_collected < SAMPLES_PER_ALGO:
                # Randomize warehouse dimensions for this sample
                # Length/Width: 5-50m, Height: 3-15m
                wh_l = round(random.uniform(5.0, 50.0), 1)
                wh_w = round(random.uniform(5.0, 50.0), 1)
                wh_h = round(random.uniform(3.0, 15.0), 1)
                
                # Update warehouse config
                warehouse = {
                    'id': 1,
                    'length': wh_l, 'width': wh_w, 'height': wh_h,
                    'door_x': 0, 'door_y': 0,
                    'layer_heights': [] 
                }
                
                items = generate_random_items(ITEMS_PER_SAMPLE, (wh_l, wh_w, wh_h))
                
                # Check method signature
                try:
                    if 'Hybrid' in str(type(optimizer)):
                        if algo_name == 'fit_eo_ga':
                            sol, fitness, _ = optimizer.optimize_eo_ga(items, warehouse, weights)
                        else:
                            sol, fitness, _ = optimizer.optimize(items, warehouse, weights)
                    else:
                        sol, fitness, _ = optimizer.optimize(items, warehouse, weights)
                except Exception as e:
                    print(f"Error running {algo_name}: {e}")
                    continue
                
                # Filter for quality - we want GOOD examples
                # Simple heuristic: if most items were placed successfully (z < 10000)
                # Note: optimizer typically sets z very high if it fails.
                # Let's count valid items.
                
                valid_items = 0
                rows_buffer = []
                
                # sol is closest to numpy array or list of dicts depending on optimizer return
                # Base optimizers usually return numpy solution array (N, 4) in current codebase
                # BUT app.py converts list of dicts. Let's check what optimize returns.
                # Reading optimizer.py: "return best_solution, best_fitness, ..."
                # optimize() usually converts to list of dicts at the end?
                # No, optimizer.py return create_random_solution_array result which is numpy (N,4).
                # Actually, let's verify optimizer.py optimize method return type.
                # GA: returns `best_population[0]` which is numpy array.
                
                if isinstance(sol, list):
                    # Convert to array-like access
                    pass 
                elif isinstance(sol, np.ndarray):
                    # It is numpy array
                    pass
                
                # Assuming numpy array (N,4)
                for i, item in enumerate(items):
                    if isinstance(sol, np.ndarray):
                         tx, ty, tz, trot = sol[i]
                    else:
                         tx, ty, tz, trot = sol[i]['x'], sol[i]['y'], sol[i]['z'], sol[i]['rotation']
                    
                    if tz > 5000: # Failed placement usually
                        continue
                        
                    valid_items += 1
                    
                    rows_buffer.append([
                         item['length'], item['width'], item['height'], item['weight'],
                         1 if item['fragility'] else 0, 1 if item['stackable'] else 0, 1 if item['can_rotate'] else 0,
                         warehouse['length'], warehouse['width'], warehouse['height'],
                         tx, ty, tz, trot
                    ])
                
                # Only save if packing was decent (> 80% items placed)
                if valid_items >= ITEMS_PER_SAMPLE * 0.8:
                    writer.writerows(rows_buffer)
                    samples_collected += 1
                    print(f"[{algo_name}] Collected sample {samples_collected}/{SAMPLES_PER_ALGO} (Fit: {fitness:.4f})")
                    f.flush()
                else:
                    # Retry
                    pass

    print("Data generation complete.")

if __name__ == "__main__":
    run_generation()
