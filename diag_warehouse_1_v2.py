import numpy as np
import sys
import os

# Add current dir to path to import optimizer, database
sys.path.append(os.getcwd())

from optimizer import repair_solution_compact
from database import get_exclusion_zones, get_all_items, get_warehouse_config

def diag():
    wh_id = 1
    warehouse = get_warehouse_config(wh_id)
    items = get_all_items(wh_id)
    zones = get_exclusion_zones(wh_id)
    alloc_zones = [z for z in zones if z['zone_type'] == 'allocation']
    
    # Setup props
    items_props = np.zeros((len(items), 9), dtype=np.float32)
    for i, item in enumerate(items):
        items_props[i] = [item['length'], item['width'], item['height'], item['can_rotate'], item['stackable'], item['access_freq'], item['weight'], 1, item['fragility']]

    solution = np.zeros((len(items), 4), dtype=np.float32)
    solution[:, 0] = 5.0; solution[:, 1] = 5.0; solution[:, 2] = 0.0

    repaired = repair_solution_compact(solution, items_props, (10, 10, 10, 0, 0), alloc_zones)

    zone_stats = {z['name']: {'count': 0, 'vol': 0.0, 'large': 0, 'capacity': (z['x2']-z['x1'])*(z['y2']-z['y1'])*(z['z2']-z['z1'])} for z in alloc_zones}

    for i in range(len(items)):
        cx, cy, cz = repaired[i, 0], repaired[i, 1], repaired[i, 2]
        item_vol = items_props[i, 0] * items_props[i, 1] * items_props[i, 2]
        for z in alloc_zones:
            if z['x1']-0.1 <= cx <= z['x2']+0.1 and z['y1']-0.1 <= cy <= z['y2']+0.1 and z['z1']-0.1 <= cz <= z['z2']+0.1:
                zone_stats[z['name']]['count'] += 1
                zone_stats[z['name']]['vol'] += item_vol
                if item_vol > 0.5:
                    zone_stats[z['name']]['large'] += 1
                break
    
    print("\nDetailed Placement Results (with size distribution):")
    for name, stats in sorted(zone_stats.items()):
        util = (stats['vol'] / stats['capacity']) * 100 if stats['capacity'] > 0 else 0
        avg_v = stats['vol'] / stats['count'] if stats['count'] > 0 else 0
        print(f"{name}: {stats['count']} items (Avg Vol: {avg_v:.2f}, >0.5 Vol: {stats['large']}), Util: {util:.1f}%")

if __name__ == "__main__":
    diag()
