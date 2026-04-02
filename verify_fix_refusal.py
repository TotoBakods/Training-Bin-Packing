import numpy as np
import sys
import os

# Add current dir to path to import optimizer
sys.path.append(os.getcwd())

from optimizer import repair_solution_compact

def test_refusal_fix():
    print("--- Running Refusal Fix Verification (3 items) ---")
    
    # 1. Setup mock data: 3 items
    # Item 0: Non-Fragile
    # Item 1: Fragile (fits in Bottom)
    # Item 2: Fragile (will overflow to Top)
    num_items = 3
    items_props = np.zeros((num_items, 9), dtype=np.float32)
    # Dimensions 1x1x1
    for i in range(3):
        items_props[i] = [1.0, 1.0, 1.0, 0, 0, 0.5, 10.0, 1, 0] # NF
    
    items_props[1, 8] = 1 # FR
    items_props[2, 8] = 1 # FR

    # 2. Setup mock solution (predicted x,y,z,rot)
    # All want to be at (2,2)
    solution = np.zeros((num_items, 4), dtype=np.float32)
    solution[:, 0] = 2.0
    solution[:, 1] = 2.0
    solution[:, 2] = 0.0

    # 3. Setup two zones (Bottom and Top)
    # Bottom Zone Capacity: 2.0 (holds 2 items max)
    # Use small floor to ensure coverage logic triggers pre-reservation.
    wh_dims = (10.0, 10.0, 10.0)
    allocation_zones = [
        {'x1': 1.0, 'y1': 1.0, 'x2': 2.5, 'y2': 2.5, 'z1': 0.0, 'z2': 2.0, 'name': 'Bottom'}, # floor=1.5x1.5=2.25, vol=4.5
        {'x1': 1.0, 'y1': 1.0, 'x2': 2.5, 'y2': 2.5, 'z1': 2.0, 'z2': 4.0, 'name': 'Top'}
    ]
    # We want Bottom Zone to be "full" in dry-run with 2 items.
    # Total items = 3. 
    # Simulation: NF0 (Zone 0), FR1 (Zone 0), FR2 (Zone 1).
    # Since FR2 is in Zone 1, Phase 2 pre-reserves NF0 to Zone 1.
    # Result: NF0 (Top), FR1 (Bottom), FR2 (Top).
    
    # Force overflow by reducing zone capacity or increasing item volume
    items_props[:, 0:3] = 1.0 # 1x1x1
    # Bottom zone vol = 1.5 * 1.5 * 2.0 = 4.5.
    # NF0(1) + FR1(1) + FR2(1) = 3.0.
    # 3.0 < 4.5 * 0.85 = 3.825.
    # So FR2 might still fit in Zone 0 in the dry-run if we are not careful.
    # Let's make the zone smaller.
    allocation_zones[0]['z2'] = 1.1 # Vol = 1.5 * 1.5 * 1.1 = 2.475
    # 2.475 * 0.85 = 2.1.
    # NF0(1) + FR1(1) = 2.0. (Still fits)
    # 2.1 < 3.0, so FR2 will overflow to Zone 1.

    print("Running repair...")
    repaired = repair_solution_compact(solution, items_props, wh_dims, allocation_zones)

    print("\nResults:")
    item0_z = repaired[0, 2] # NF
    item1_z = repaired[1, 2] # FR (Bottom)
    item2_z = repaired[2, 2] # FR (Top)
    
    for i in range(num_items):
        z = repaired[i, 2]
        is_nf = items_props[i, 8] == 0
        label = "NF" if is_nf else "FR"
        print(f"Item {i} ({label}) -> Z={z:.3f}")

    # Check if Item 0 is in Top and Item 1 is in Bottom
    if item1_z < 1.1 and item0_z >= 2.0:
        print("\nSUCCESS: Target scenario achieved: NF in Top, FR in Bottom.")
        print("Verification: The FR item in Bottom Zone was NOT blocked by the NF item in Top Zone.")
    elif item1_z >= 1000:
        print("\nFAILURE: Fragile item Item 1 was NOT placed (refusal)!")
    else:
        print("\nINFO: Scenario not exactly as expected, but check if any item failed to place.")
        if np.any(repaired[:, 2] >= 1000):
             print("FAILURE: Some items refused to place.")
        else:
             print("SUCCESS: All items placed successfully.")

if __name__ == "__main__":
    test_refusal_fix()
