import numpy as np
import sys
import os

# Add current dir to path to import optimizer
sys.path.append(os.getcwd())

from optimizer import repair_solution_compact

def test_deferral():
    print("--- Running Multi-Zone Deferral Test ---")
    
    # 1. Setup mock data
    # 60 items total: 50 non-fragile, 10 fragile
    num_items = 60
    items_props = np.zeros((num_items, 9), dtype=np.float32)
    # Dimensions 1x1x1
    for i in range(num_items):
        items_props[i] = [1.0, 1.0, 1.0, 1, 1, 0.5, 10.0, 1, 0] # non-fragile
    for i in range(50, 60):
        items_props[i, 8] = 1 # fragile

    # 2. Setup mock solution (predicted x,y,z,rot)
    solution = np.zeros((num_items, 4), dtype=np.float32)
    solution[:, 0] = 2.0
    solution[:, 1] = 2.0
    solution[:, 2] = 0.0

    # 3. Setup two zones (Bottom and Top)
    # We want Bottom Zone to be able to hold ALL Non-Fragile items (50),
    # but NOT all items (60).
    # Bottom Zone Capacity: 5x5x2.5 = 62.5
    # Total volume = 60.
    # Non-Fragile volume = 50.
    # Bottom Zone can hold all 50 NF, but not 50 NF + 10 F (60 is near 62.5 * 0.8 efficiency)
    wh_dims = (10.0, 10.0, 10.0, 0.0, 0.0)
    allocation_zones = [
        {'x1': 1.0, 'y1': 1.0, 'x2': 6.0, 'y2': 6.0, 'z1': 0.0, 'z2': 2.5, 'zone_type': 'allocation'}, # Bottom
        {'x1': 1.0, 'y1': 1.0, 'x2': 6.0, 'y2': 6.0, 'z1': 2.5, 'z2': 5.0, 'zone_type': 'allocation'}  # Top
    ]

    # 4. Run repair
    repaired = repair_solution_compact(solution, items_props, wh_dims, allocation_zones)

    # 5. Check results
    zone0_nf = 0
    zone0_f = 0
    zone1_nf = 0
    zone1_f = 0
    
    for i in range(num_items):
        z = repaired[i, 2]
        is_fragile = items_props[i, 8] == 1
        if 0.0 <= z < 2.5:
            if is_fragile: zone0_f += 1
            else: zone0_nf += 1
        elif 2.5 <= z < 5.0:
            if is_fragile: zone1_f += 1
            else: zone1_nf += 1
        elif z >= 1000:
            print(f"Item {i} overflowed completely (Z={z})")

    print(f"Zone 0 (Bottom) - Non-Fragile: {zone0_nf}, Fragile: {zone0_f}")
    print(f"Zone 1 (Top)    - Non-Fragile: {zone1_nf}, Fragile: {zone1_f}")
    
    # Validation
    if zone1_nf > 0:
        print("SUCCESS: Non-fragile items deferred to Top Zone.")
    else:
        print("FAILURE: No non-fragile items in Top Zone despite fragile presence.")

    if zone1_f > 0:
        # Check if any fragile item in Zone 1 is on the floor (Z=2.5)
        # We expect them to be on top of the deferred non-fragile items.
        fragile_on_floor = 0
        for i in range(50, 60):
            if 2.5 <= repaired[i, 2] < 2.6:
                fragile_on_floor += 1
        
        if fragile_on_floor == 0:
             print("SUCCESS: Fragile items in Top Zone are supported by non-fragile items.")
        else:
             print(f"WARNING: {fragile_on_floor} fragile items in Top Zone are directly on the zone floor.")

if __name__ == "__main__":
    test_deferral()
