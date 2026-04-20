import numpy as np
import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

from optimizer import repair_solution_compact

def test_sorting():
    # 4 items:
    # 0: Low AF, Low Prio, Non-Fragile
    # 1: High AF, Low Prio, Non-Fragile
    # 2: Low AF, High Prio, Non-Fragile
    # 3: High AF, High Prio, Fragile
    
    # props: [l, w, h, can_rot, stackable, af, weight, cat, fragility, priority]
    items_props = np.array([
        [1, 1, 1, 1, 1, 2, 10, 1, 0, 1], # Item 0: AF=2, Prio=1, NF
        [1, 1, 1, 1, 1, 10, 10, 1, 0, 1], # Item 1: AF=10, Prio=1, NF
        [1, 1, 1, 1, 1, 2, 10, 1, 0, 5], # Item 2: AF=2, Prio=5, NF
        [1, 1, 1, 1, 1, 10, 10, 1, 1, 5], # Item 3: AF=10, Prio=5, FR
    ], dtype=np.float32)
    
    solution = np.zeros((4, 4), dtype=np.float32)
    warehouse_dims = (10, 10, 10)
    
    print("Running repair...")
    repaired = repair_solution_compact(solution, items_props, warehouse_dims)
    
    print("\nCheck placement_debug.log for PASS-A and PASS-C order.")
    
if __name__ == "__main__":
    test_sorting()
