import torch
import torch.nn as nn
import os
import numpy as np
import time
from database import get_exclusion_zones
from optimizer import (
    repair_solution_compact, 
    fitness_function_numpy, 
    get_valid_z_positions
)
import platform
import psutil

def get_system_metadata():
    """Captures hardware and software environment details for documentation."""
    metadata = {
        "os": platform.system(),
        "os_release": platform.release(),
        "cpu_name": platform.processor(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": platform.python_version(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    return metadata

class PackingModel(nn.Module):
    """
    Neural network for predicting normalized (x, y, z, rot) placement coordinates.
    Outputs are constrained to [0, 1] via Sigmoid to match normalized target range.
    """
    def __init__(self, input_dim=19, output_dim=4):
        super(PackingModel, self).__init__()
        self.net = nn.Sequential(
            # Layer 1: Condensed Input
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            # Layer 2: Feature Extraction
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            # Layer 3: Dimensional Narrowing
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            
            # Output: Scaled to [0, 1] for unit-space coordinate prediction
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class MLOptimizer:
    """Uses trained Neural Network models to predict item positions."""
    def __init__(self, variant="fit_ga"):
        # Map human-readable or API-level variant names to model filenames
        variant_map = {
            "ga": "fit_ga",
            "eo": "fit_eo",
            "ga_eo": "fit_ga_eo",
            "eo_ga": "fit_eo_ga",
            "fit_ga": "fit_ga",
            "fit_eo": "fit_eo",
            "fit_ga_eo": "fit_ga_eo",
            "fit_eo_ga": "fit_eo_ga"
        }
        self.variant = variant_map.get(variant.lower(), "fit_ga")
        self.model_name = self.variant
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
             # Fallback to standard GA if model missing has been removed
             print("ML Model missing, cannot proceed.")
             return [], 0, 0

        # Pre-process items
        zones = get_exclusion_zones(warehouse['id'])
        exclusion_zones_arr = None
        if zones:
             ex_zones = [z for z in zones if z['zone_type'] == 'exclusion']
             if ex_zones:
                 exclusion_zones_arr = np.array([[z['x1'], z['y1'], z['x2'], z['y2']] for z in ex_zones])
        
        # Props for repair
        items_props = np.zeros((num_items, 9), dtype=np.float32)
        
        # Features for Model: 19 advanced geometric features (including Sequence Progress)
        features = np.zeros((num_items, 19), dtype=np.float32)
        
        wh_l = warehouse['length']
        wh_w = warehouse['width']
        wh_h = warehouse['height']
        
        # Pre-calculate warehouse values
        wh_vol = wh_l * wh_w * wh_h
        wh_area = wh_l * wh_w
        
        for i, item in enumerate(items):
            # Props (for heuristic)
            items_props[i] = [
                item['length'], item['width'], item['height'],
                item['can_rotate'], item['stackable'],
                item['access_freq'], item.get('weight', 0),
                hash(item.get('category', '')) % 10000,
                item.get('fragility', 0)
            ]
            
            # Features (for Neural Network)
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
                1.0 if item.get('can_rotate', 1) else 0.0,
                wh_l / 100.0,
                wh_w / 100.0,
                wh_h / 100.0,
                # Advanced features
                item_vol / 10.0,
                wh_vol / 1000.0,
                item_vol / (wh_vol + 1e-6),
                item_area / 10.0,
                wh_area / 100.0,
                item_area / (wh_area + 1e-6),
                l / (wh_l + 1e-6),
                w / (wh_w + 1e-6),
                # 19. Sequence Progress (Proxy for fill level)
                i / float(num_items)
            ]

        # Inference
        try:
            inference_start = time.time()
            with torch.no_grad():
                inputs = torch.tensor(features).to(self.device)
                outputs = self.model(inputs) # (N, 4) -> x, y, z, rot
                outputs = outputs.cpu().numpy()
            inference_end = time.time()
            inf_latency = (inference_end - inference_start) * 1000 # ms
                
            # Denormalize
            pred_x = outputs[:, 0] * wh_l
            pred_y = outputs[:, 1] * wh_w
            pred_z = outputs[:, 2] * wh_h
            # Clamp Z > 0
            pred_z = np.maximum(pred_z, 0)
            pred_rot = outputs[:, 3] * 6.0

            # Store ML predictions for displacement calculation
            ml_predictions = np.column_stack((pred_x, pred_y, pred_z, pred_rot))
            
            # Build initial solution array from ML predictions
            solution = np.column_stack((pred_x, pred_y, pred_z, pred_rot))
            
            # Repair (Physics & Constraints)
            valid_z = get_valid_z_positions(warehouse)
            allocation_zones = None
            if zones:
                alloc_zones = [z for z in zones if z['zone_type'] == 'allocation']
                if alloc_zones:
                    allocation_zones = alloc_zones
            
            # Repair using compact logic
            is_eo_ga = self.model_name == "fit_eo_ga"
            repair_start = time.time()
            if callback:
                def intermediate_callback(intermediate_sol):
                    # Convert numpy array to list of dicts for real-time updates
                    intermediate_list = []
                    for i in range(num_items):
                        intermediate_list.append({
                            'id': items[i]['id'],
                            'x': float(intermediate_sol[i, 0]),
                            'y': float(intermediate_sol[i, 1]),
                            'z': float(intermediate_sol[i, 2]),
                            'rotation': int(intermediate_sol[i, 3])
                        })
                    callback(50, 0, 0, intermediate_list, 0, 0, 0, message="Repairing layout (Tetris Style)...")
                
                callback(20, 0, 0, None, 0, 0, 0, message="ML Inference complete. Applying Physics Settlement...")
                
                # Higher callback interval for fast_mode to reduce JSON/List conversion overhead
                cb_interval = 200 if is_eo_ga else 50
                solution = repair_solution_compact(
                    solution, items_props, (wh_l, wh_w, wh_h, 0, 0), allocation_zones, valid_z, 
                    callback=intermediate_callback, callback_interval=cb_interval, fast_mode=is_eo_ga
                )
            else:
                solution = repair_solution_compact(
                    solution, items_props, (wh_l, wh_w, wh_h, 0, 0), allocation_zones, valid_z, 
                    fast_mode=is_eo_ga
                )
            repair_end = time.time()
            repair_latency = (repair_end - repair_start) * 1000 # ms
            
            if callback:
                callback(80, 0, 0, None, 0, 0, 0, message="Physics Settlement Complete.")
            
            # Calculate Fitness
            current_weights = weights if weights else {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1}
            fitness, su, acc, sta, grp = fitness_function_numpy(
                solution, items_props, (wh_l, wh_w, wh_h, 0, 0), current_weights, valid_z, exclusion_zones_arr
            )
            
            # --- Inference Metrics / Diagnostics ---
            # Displacement: Euclidean distance between ML guess and Heuristic final
            # Only count items that were actually placed (Z < 1000)
            placed_mask = solution[:, 2] < 1000
            if np.any(placed_mask):
                sq_diff = (solution[placed_mask, :3] - ml_predictions[placed_mask, :3])**2
                avg_displacement = np.mean(np.sqrt(np.sum(sq_diff, axis=1)))
            else:
                avg_displacement = 0.0
            
            # Volumetric Efficiency: Actual item volume / Bounding Box volume of placed group
            if np.any(placed_mask):
                placed_vol = np.sum(items_props[placed_mask, 0] * items_props[placed_mask, 1] * items_props[placed_mask, 2])
                min_c = np.min(solution[placed_mask, :3], axis=0)
                max_c = np.max(solution[placed_mask, :3] + items_props[placed_mask, 0:3], axis=0) # rough approx of max
                bbox_vol = np.prod(max_c - min_c)
                vol_eff = placed_vol / (bbox_vol + 1e-6)
            else:
                vol_eff = 0.0

            # --- SOTA Metric Validation (PSR and SSR) ---
            # PSR: Placement Success Rate (items within absolute bounds)
            success_count = np.sum(
                (solution[:, 0] >= 0) & (solution[:, 0] + items_props[:, 0] <= wh_l + 0.05) &
                (solution[:, 1] >= 0) & (solution[:, 1] + items_props[:, 1] <= wh_w + 0.05) &
                (solution[:, 2] >= 0) & (solution[:, 2] + items_props[:, 2] <= wh_h + 0.05)
            )
            psr = (success_count / num_items) * 100.0

            # SSR: Support Surface Ratio (Area supported by objects below / Floor)
            # This is a bit expensive, so we use a vectorized approach relative to placed items
            ssr_values = []
            for i in range(num_items):
                if solution[i, 2] <= 0.01:
                    ssr_values.append(100.0)
                    continue
                
                # Check overlap with items below
                mask_below = (np.abs((solution[:, 2] + items_props[:, 2]) - solution[i, 2]) < 0.05)
                if not np.any(mask_below):
                    ssr_values.append(0.0)
                    continue
                
                # XY Overlap Calculation (Vectorized for those below)
                ix1 = np.maximum(solution[i, 0], solution[mask_below, 0])
                ix2 = np.minimum(solution[i, 0] + items_props[i, 0], solution[mask_below, 0] + items_props[mask_below, 0])
                iy1 = np.maximum(solution[i, 1], solution[mask_below, 1])
                iy2 = np.minimum(solution[i, 1] + items_props[i, 1], solution[mask_below, 1] + items_props[mask_below, 1])
                
                overlaps = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
                total_supp_area = np.sum(overlaps)
                item_area = items_props[i, 0] * items_props[i, 1]
                ssr_values.append(min(1.0, total_supp_area / (item_area + 1e-6)) * 100.0)
            
            avg_ssr = np.mean(ssr_values) if ssr_values else 0.0

            metrics = {
                "inference_latency_ms": float(inf_latency),
                "repair_latency_ms": float(repair_latency),
                "total_latency_ms": float(inf_latency + repair_latency),
                "avg_displacement": float(avg_displacement),
                "volumetric_efficiency": float(vol_eff),
                "placed_count": int(np.sum(placed_mask)),
                "psr_pct": float(psr),
                "ssr_pct": float(avg_ssr),
                "variant": self.model_name,
                "is_fast_path": is_eo_ga
            }
            
            time_to_best = time.time() - start_time
            
            # Convert
            final_sol_list = []
            for i in range(num_items):
                final_sol_list.append({
                    'id': items[i]['id'],
                    'x': float(solution[i, 0]),
                    'y': float(solution[i, 1]),
                    'z': float(solution[i, 2]),
                    'rotation': int(solution[i, 3]),
                    'ml_x': float(ml_predictions[i, 0]),
                    'ml_y': float(ml_predictions[i, 1]),
                    'ml_z': float(ml_predictions[i, 2])
                })
                
            return final_sol_list, float(fitness), time_to_best, metrics
            
        except Exception as e:
            print(f"ML Inference Error: {e}")
            import traceback
            traceback.print_exc()
            return [], 0, 0, {}
