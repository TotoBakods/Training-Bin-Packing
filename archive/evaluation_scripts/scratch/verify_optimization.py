import numpy as np
import torch
from optimizer import repair_solution_compact

def test_full_optimization():
    print("\n--- Verifying Force Balance (AF vs Priority) ---")
    # Warehouse: 10x10x10, Door at (0,0)
    wh_dims = [10.0, 10.0, 10.0, 0.0, 0.0]
    
    # Items:
    # 0: High AF (10), Low Prio (1) -> Should be near Door (0,0)
    # 1: Low AF (0), High Prio (10) -> Should be near a Wall but not necessarily near Door
    # 2: Mid AF (5), Mid Prio (5) -> Should be near Door AND near a Wall
    items_props = np.array([
        [1, 1, 1, 0, 1, 10, 10, 0, 0, 1], # 0: AF=10, Prio=1
        [1, 1, 1, 0, 1, 0,  10, 0, 0, 10], # 1: AF=0,  Prio=10
        [1, 1, 1, 0, 1, 5,  10, 0, 0, 5],  # 2: AF=5,  Prio=5
    ], dtype=np.float32)
    
    solution = np.array([
        [5, 5, 0, 0],
        [5, 5, 0, 0],
        [5, 5, 0, 0],
    ], dtype=np.float32)
    
    final_sol = repair_solution_compact(solution, items_props, wh_dims)
    
    print(f"Item 0 (AF=10, Prio=1) final pos: {final_sol[0, 0:3]}")
    print(f"Item 1 (AF=0, Prio=10) final pos: {final_sol[1, 0:3]}")
    print(f"Item 2 (AF=5, Prio=5) final pos: {final_sol[2, 0:3]}")

    print("\n--- Verifying Lightweight Fragile Stacking ---")
    # Warehouse: 1.1x1.1x10 (forces stacking)
    wh_dims_s = [1.1, 1.1, 10.0, 0.0, 0.0]
    # 2 Items, both fragile. One heavy (100kg), one light (1kg).
    # Heavy one should be at Z=0, light one at Z=1.
    items_props_f = np.array([
        [1, 1, 1, 0, 1, 0, 100, 0, 1, 1], # 0: Fragile, 100kg
        [1, 1, 1, 0, 1, 0, 1,   0, 1, 1], # 1: Fragile, 1kg
    ], dtype=np.float32)
    
    solution_f = np.array([
        [0.55, 0.55, 0, 0],
        [0.55, 0.55, 0, 0],
    ], dtype=np.float32)
    
    final_sol_f = repair_solution_compact(solution_f, items_props_f, wh_dims_s)
    
    print(f"Item 0 (100kg FR) final pos: {final_sol_f[0, 0:3]}")
    print(f"Item 1 (1kg FR) final pos: {final_sol_f[1, 0:3]}")
    
    if final_sol_f[1, 2] > final_sol_f[0, 2]:
        print("SUCCESS: Lightweight fragile item placed on top of heavy fragile item.")
    else:
        print("FAILURE: Lightweight fragile item NOT on top.")

if __name__ == "__main__":
    test_full_optimization()
