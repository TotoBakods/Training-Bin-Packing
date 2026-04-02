import numpy as np
import time
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from optimizer import repair_solution_compact

def run_benchmark(profile_name, bin_size, item_max_ratio, num_items):
    print(f"\n--- SOTA REPLICATION: {profile_name} ---")
    print(f"Config: Bin {bin_size}^3, Items <= {item_max_ratio*100}% Bin Dim, Count: {num_items}")
    
    wh_dims = (bin_size, bin_size, bin_size)
    alloc_zones = [{'name': 'Bin', 'x1': 0.0, 'y1': 0.0, 'z1': 0.0, 'x2': float(bin_size), 'y2': float(bin_size), 'z2': float(bin_size), 'zone_type': 'allocation'}]
    
    # Generate items
    items_props = np.zeros((num_items, 9), dtype=np.float32)
    for i in range(num_items):
        it_dims = np.random.uniform(1.0, bin_size * item_max_ratio, 3)
        items_props[i] = [it_dims[0], it_dims[1], it_dims[2], 1, 1, 1, 10.0, 1, 0]
    
    # Random initial placements (Tetris-style)
    solution = np.zeros((num_items, 4), dtype=np.float32)
    solution[:, 0] = bin_size / 2.0; solution[:, 1] = bin_size / 2.0; solution[:, 2] = 0.0
    
    start = time.time()
    repaired = repair_solution_compact(solution, items_props, (wh_dims[0], wh_dims[1], wh_dims[2], 0, 0), alloc_zones, fast_mode=True)
    lat = (time.time() - start) * 1000 / num_items
    
    success = 0
    vol = 0
    ssr_sum = 0
    for i in range(num_items):
        if repaired[i, 2] < 1000:
            success += 1
            vol += items_props[i, 0] * items_props[i, 1] * items_props[i, 2]
            
            if repaired[i, 2] <= 0.05:
                ssr_sum += 1.0
            else:
                mask_below = (np.abs((repaired[:, 2] + items_props[:, 2]) - repaired[i, 2]) < 0.05)
                if np.any(mask_below):
                    ix1 = np.maximum(repaired[i, 0], repaired[mask_below, 0])
                    ix2 = np.minimum(repaired[i, 0] + items_props[i, 0], repaired[mask_below, 0] + items_props[mask_below, 0])
                    iy1 = np.maximum(repaired[i, 1], repaired[mask_below, 1])
                    iy2 = np.minimum(repaired[i, 1] + items_props[i, 1], repaired[mask_below, 1] + items_props[mask_below, 1])
                    overlaps = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
                    ssr_sum += min(1.0, np.sum(overlaps) / (items_props[i, 0] * items_props[i, 1] + 1e-6))
    
    psr = (success / num_items) * 100.0
    ssr = (ssr_sum / max(1, success)) * 100.0
    vu = (vol / (bin_size**3)) * 100.0
    
    print(f"RESULT >> PSR: {psr:.1f}% | SSR: {ssr:.1f}% | VU: {vu:.1f}% | LAT: {lat:.2f}ms")
    return psr, ssr, vu, lat

if __name__ == "__main__":
    # Zhao (AAAI-21): 10x10x10, items <= 5.0 (50%)
    z_psr, z_ssr, z_vu, z_lat = run_benchmark("Zhao et al. (AAAI-21)", 10.0, 0.5, 60)
    
    # Ha (PRL-17): 20x20x20, items <= 8.0 (40% of 20)
    h_psr, h_ssr, h_vu, h_lat = run_benchmark("Ha et al. (2017)", 20.0, 0.4, 250)
    
    print("\n" + "="*60)
    print("FINAL MULTI-PROFILE COMPETITIVE SCORECARD")
    print("="*60)
    print(f"{'Metric':<15} | {'Zhao Run':<12} | {'Ha Run':<12} | {'SOTA Targets':<15}")
    print("-" * 60)
    print(f"{'PSR':<15} | {z_psr:>11.1f}% | {h_psr:>11.1f}% | 94-98%")
    print(f"{'SSR':<15} | {z_ssr:>11.1f}% | {h_ssr:>11.1f}% | 68-70%")
    print(f"{'VU':<15} | {z_vu:>11.1f}% | {h_vu:>11.1f}% | 75-76%")
    print(f"{'LAT (ms)':<15} | {z_lat:>11.2f}  | {h_lat:>11.2f}  | ~25-120ms")
    print("="*60)
