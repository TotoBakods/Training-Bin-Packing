import numpy as np
import sys
import os

# Add current dir to path to import optimizer
sys.path.append(os.getcwd())

from optimizer import repair_solution_compact

def test_n_zones():
    print("--- Running Multi-Zone N-Shelf Test ---")
    
    # 60 items total: 40 non-fragile, 20 fragile
    num_items = 60
    items_props = np.zeros((num_items, 9), dtype=np.float32)
    for i in range(num_items):
        items_props[i] = [1.0, 1.0, 1.0, 1, 1, 0.5, 10.0, 1, 0] # non-fragile
    for i in range(40, 60):
        items_props[i, 8] = 1 # fragile

    solution = np.zeros((num_items, 4), dtype=np.float32)
    solution[:, 0] = 1.1; solution[:, 1] = 1.1; solution[:, 2] = 0.0

    wh_dims = (10, 10, 10, 0, 0)
    # 3 shelves: A, B, C (all at bottom)
    allocation_zones = [
        {'id': 1, 'name': 'Shelf A', 'x1': 1, 'y1': 1, 'x2': 3.1, 'y2': 3.1, 'z1': 0, 'z2': 5, 'zone_type': 'allocation'}, # 2x2 section
        {'id': 2, 'name': 'Shelf B', 'x1': 4, 'y1': 4, 'x2': 6.1, 'y2': 6.1, 'z1': 0, 'z2': 5, 'zone_type': 'allocation'}, # 2x2 section
        {'id': 3, 'name': 'Shelf C', 'x1': 7, 'y1': 7, 'x2': 9.1, 'y2': 9.1, 'z1': 0, 'z2': 5, 'zone_type': 'allocation'}  # 2x2 section
    ]

    repaired = repair_solution_compact(solution, items_props, wh_dims, allocation_zones)

    zone_counts = {1: 0, 2: 0, 3: 0, 'None': 0}
    for i in range(num_items):
        cx, cy, z = repaired[i, 0], repaired[i, 1], repaired[i, 2]
        # repaired yields center coords, but repair_solution_compact stores (cx, cy, z, rot, dx, dy, dz) 
        # wait, repair_solution_compact returns the passed solution array modified.
        # solution[idx] = [b_x, b_y, b_z, b_rot] 
        # b_x, b_y are centers.
        
        found = False
        for zne in allocation_zones:
            # Check if center is within zone bounds (with small epsilon)
            if zne['x1'] - 0.1 <= cx <= zne['x2'] + 0.1 and zne['y1'] - 0.1 <= cy <= zne['y2'] + 0.1:
                zone_counts[zne['id']] += 1
                found = True
                break
        if not found: zone_counts['None'] += 1

    print(f"Results: {zone_counts}")
    if zone_counts[3] > 0:
        print("SUCCESS: Shelf C received items.")
    else:
        print("FAILURE: Shelf C is still empty.")
        
    # Also verify structural stability: ALL NF from any zone should have smaller Z than F from any zone?
    # No, that's not strictly true if they are in different shelves.
    # But for ANY zone, NF should be below F.
    
    violations = 0
    for zne in allocation_zones:
        z_items = []
        for i in range(num_items):
            cx, cy = repaired[i, 0], repaired[i, 1]
            if zne['x1'] - 0.1 <= cx <= zne['x2'] + 0.1 and zne['y1'] - 0.1 <= cy <= zne['y2'] + 0.1:
                z_items.append((repaired[i, 2], items_props[i, 8])) # (Z, is_fragile)
        
        # Sort by Z
        z_items.sort()
        # Ensure no fragile is below a non-fragile in the same zone
        seen_fragile = False
        for z, fragile in z_items:
            if fragile == 1:
                seen_fragile = True
            elif fragile == 0 and seen_fragile:
                violations += 1
                
    if violations == 0:
        print("SUCCESS: Structural stability (NF below F) maintained in all shelves.")
    else:
        print(f"FAILURE: {violations} structural violations detected.")

if __name__ == "__main__":
    test_n_zones()
