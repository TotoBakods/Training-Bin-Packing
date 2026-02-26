import torch
import torch.nn as nn
import os
import numpy as np
import time
from database import get_exclusion_zones
from optimizer import (
    GeneticAlgorithm, 
    repair_solution_compact, 
    fitness_function_numpy, 
    get_valid_z_positions
)

class PackingModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=4):
        super(PackingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)
        return out

class MLOptimizer:
    """Uses trained Neural Network models to predict item positions."""
    def __init__(self, model_name="fit_ga"):
        self.model_name = model_name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()

    def _load_model(self):
        try:
             model_path = os.path.join("models", f"model_{self.model_name}.pth")
             if not os.path.exists(model_path):
                 print(f"Model {model_path} not found.")
                 return
             
             self.model = PackingModel()
             self.model.load_state_dict(torch.load(model_path, map_location=self.device))
             self.model.to(self.device)
             self.model.eval()
             print(f"Loaded ML model: {model_path}")
        except Exception as e:
            print(f"Failed to load ML model: {e}")

    def optimize(self, items, warehouse, weights=None, callback=None, optimization_state=None):
        num_items = len(items)
        if num_items == 0:
            return [], 0, 0
            
        start_time = time.time()
        
        if self.model is None:
             # Fallback to standard GA if model missing
             print("ML Model missing, falling back to GA")
             ga = GeneticAlgorithm(generations=50, population_size=50) 
             return ga.optimize(items, warehouse, weights, callback, optimization_state)

        # Pre-process items
        zones = get_exclusion_zones(warehouse['id'])
        exclusion_zones_arr = None
        if zones:
             ex_zones = [z for z in zones if z['zone_type'] == 'exclusion']
             if ex_zones:
                 exclusion_zones_arr = np.array([[z['x1'], z['y1'], z['x2'], z['y2']] for z in ex_zones])
        
        # Props for repair
        items_props = np.zeros((num_items, 9), dtype=np.float32)
        
        # Features for Model: l, w, h, weight, fragile, stackable, can_rotate, wh_l, wh_w, wh_h
        # (N, 10)
        features = np.zeros((num_items, 10), dtype=np.float32)
        
        wh_l = warehouse['length']
        wh_w = warehouse['width']
        wh_h = warehouse['height']
        
        for i, item in enumerate(items):
            # Props
            items_props[i] = [
                item['length'], item['width'], item['height'],
                item['can_rotate'], item['stackable'],
                item['access_freq'], item.get('weight', 0),
                hash(item.get('category', '')) % 10000,
                item.get('fragility', 0)
            ]
            
            # Features
            features[i] = [
                item['length'] / 10.0, 
                item['width'] / 10.0, 
                item['height'] / 10.0,
                item.get('weight', 0) / 10.0, 
                1.0 if item.get('fragility', 0) else 0.0,
                1.0 if item.get('stackable', 1) else 0.0,
                1.0 if item.get('can_rotate', 1) else 0.0,
                wh_l / 100.0,
                wh_w / 100.0,
                wh_h / 100.0
            ]

        # Inference
        try:
            with torch.no_grad():
                inputs = torch.tensor(features).to(self.device)
                outputs = self.model(inputs) # (N, 4) -> x, y, z, rot
                outputs = outputs.cpu().numpy()
                
            # Denormalize
            pred_x = outputs[:, 0] * wh_l
            pred_y = outputs[:, 1] * wh_w
            pred_z = outputs[:, 2] * wh_h
            # Clamp Z > 0
            pred_z = np.maximum(pred_z, 0)
            
            pred_rot = outputs[:, 3] * 6.0
            
            solution = np.column_stack((pred_x, pred_y, pred_z, pred_rot))
            
            # Repair (Physics & Constraints)
            valid_z = get_valid_z_positions(warehouse)
            allocation_zones = None
            if zones:
                alloc_zones = [z for z in zones if z['zone_type'] == 'allocation']
                if alloc_zones:
                    allocation_zones = alloc_zones
            
            if callback:
                callback(20, 0, 0, None, 0, 0, 0, message="ML Inference complete. Applying Physics Settlement (Tetris Style)...")
            
            # Repair using compact logic
            solution = repair_solution_compact(solution, items_props, (wh_l, wh_w, wh_h, 0, 0), allocation_zones, valid_z)
            
            if callback:
                callback(80, 0, 0, None, 0, 0, 0, message="Physics Settlement Complete.")
            
            # Calculate Fitness
            current_weights = weights if weights else {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1}
            fitness, su, acc, sta, grp = fitness_function_numpy(
                solution, items_props, (wh_l, wh_w, wh_h, 0, 0), current_weights, valid_z, exclusion_zones_arr
            )
            
            time_to_best = time.time() - start_time
            
            # Convert
            final_sol_list = []
            for i in range(num_items):
                final_sol_list.append({
                    'id': items[i]['id'],
                    'x': float(solution[i, 0]),
                    'y': float(solution[i, 1]),
                    'z': float(solution[i, 2]),
                    'rotation': int(solution[i, 3])
                })
                
            return final_sol_list, float(fitness), time_to_best
            
        except Exception as e:
            print(f"ML Inference Error: {e}")
            import traceback
            traceback.print_exc()
            return [], 0, 0
