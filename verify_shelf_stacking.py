import numpy as np
from optimizer import repair_solution_compact

def test_stacking(load_name, items_count, nf_fr_ratio=0.5):
    print(f"\n--- Testing {load_name} Stacking ({items_count} items) ---")
    
    # 1. Setup simple 2-shelf warehouse
    wh_l, wh_w, wh_h = 10.0, 10.0, 6.0
    warehouse_dims = (wh_l, wh_w, wh_h, 0, 0)
    
    # 2. Define 2 zones (Bottom: 0-3, Top: 3-6)
    allocation_zones = [
        {'x1':0, 'y1':0, 'x2':10, 'y2':10, 'z1':0, 'z2':3, 'id':0},
        {'x1':0, 'y1':0, 'x2':10, 'y2':10, 'z1':3, 'z2':6, 'id':1}
    ]
    
    # 3. Create items
    num_nf = int(items_count * nf_fr_ratio)
    num_fr = items_count - num_nf
    
    # Simple items: 1x1x1
    items_props = np.zeros((items_count, 9))
    items_props[:, 0:3] = 1.0  # l, w, h
    items_props[:, 3:6] = 1.0  # can_rotate, stackable, access_freq
    indices = np.arange(items_count)
    fragility = np.zeros(items_count)
    fragility[num_nf:] = 1.0   # Last half are fragile
    items_props[:, 8] = fragility
    
    volumes = np.ones(items_count)
    weights = np.ones(items_count) * 10.0
    
    # Dummy solution (all at 0,0,0)
    initial_solution = np.zeros((items_count, 4))
    
    # 4. Run repair
    final_sol = repair_solution_compact(
        initial_solution, items_props, warehouse_dims, allocation_zones,
        fast_mode=True
    )
    
    # 5. Analyze results
    z_coords = final_sol[:, 2]
    nf_z = z_coords[:num_nf]
    fr_z = z_coords[num_nf:]
    
    print(f"NF items in Top Shelf (z>=3): {np.sum(nf_z >= 3)}")
    print(f"FR items in Top Shelf (z>=3): {np.sum(fr_z >= 3)}")
    print(f"NF items in Bottom Shelf (z<3): {np.sum(nf_z < 3)}")
    print(f"FR items in Bottom Shelf (z<3): {np.sum(fr_z < 3)}")
    
    # Check if Top shelf HAS NF base
    if np.sum(nf_z >= 3) > 0:
        print("SUCCESS: Top shelf has Non-Fragile base layer.")
    else:
        print("WARNING: Top shelf is exclusively Fragile/Empty.")

if __name__ == "__main__":
    # Test High Load (100% capacity = 600 items in 10x10x6 with 1x1x1 items)
    # Actually 10*10*6 = 600 items. Let's use 500 to be safe.
    test_stacking("High Load", 500)
    
    # Test Low Load (20% capacity = 100 items)
    test_stacking("Low Load", 100)
