import numpy as np
import sys
import os

# Add current dir to path to import optimizer
sys.path.append(os.getcwd())

from optimizer import repair_solution_compact

def test_prio_sep():
    print("--- Verifying Separated AF (Door) vs Priority (Wall) ---")
    
    # 3 items total:
    # Item 0: High AF (10), Low Prio (1) -> Should be at door, maybe not wall
    # Item 1: Low AF (0), High Prio (3)  -> Should be at wall, maybe not door
    # Item 2: Low AF (0), Low Prio (1)   -> Should be whatever's left
    
    num_items = 3
    items_props = np.zeros((num_items, 10), dtype=np.float32)
    
    # [l, w, h, rot, stack, af, wt, cat, frag, prio]
    items_props[0] = [1.0, 1.0, 1.0, 1, 1, 10, 10, 0, 0, 1] 
    items_props[1] = [1.0, 1.0, 1.0, 1, 1, 0,  10, 0, 0, 3] 
    items_props[2] = [1.0, 1.0, 1.0, 1, 1, 0,  10, 0, 0, 1] 

    solution = np.zeros((num_items, 4), dtype=np.float32)
    wh_dims = (10, 10, 10, 0, 0) # Door at 0,0

    log_path = 'placement_debug.log'
    if os.path.exists(log_path): os.remove(log_path)

    repaired = repair_solution_compact(solution, items_props, wh_dims)

    with open(log_path, 'r') as f:
        for line in f:
            if 'PASS-A' in line: print(line.strip())

if __name__ == "__main__":
    test_prio_sep()
