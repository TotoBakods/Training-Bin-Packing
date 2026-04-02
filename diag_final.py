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

    vols = items_props[:,0] * items_props[:,1] * items_props[:,2]
    
    solution = np.zeros((len(items), 4), dtype=np.float32)
    solution[:, 0] = 5.0; solution[:, 1] = 5.0; solution[:, 2] = 0.0

    repaired = repair_solution_compact(solution, items_props, (10, 10, 10, 0, 0), alloc_zones)

    zone_stats = {z['name']: {'count': 0, 'vol': 0.0, 'large': 0, 'top_vols': []} for z in alloc_zones}

    for i in range(len(items)):
        cx, cy, cz = repaired[i, 0], repaired[i, 1], repaired[i, 2]
        item_vol = vols[i]
        for z in alloc_zones:
            if z['x1']-0.1 <= cx <= z['x2']+0.1 and z['y1']-0.1 <= cy <= z['y2']+0.1 and z['z1']-0.1 <= cz <= z['z2']+0.1:
                zone_stats[z['name']]['count'] += 1
                zone_stats[z['name']]['vol'] += item_vol
                zone_stats[z['name']]['top_vols'].append(item_vol)
                if item_vol > 0.5:
                    zone_stats[z['name']]['large'] += 1
                break
    
    print("\nFINAL DIAGNOSTICS:")
    for name, stats in sorted(zone_stats.items()):
        vols_list = sorted(stats['top_vols'], reverse=True)
        avg_vol = stats['vol']/stats['count'] if stats['count'] > 0 else 0
        print(f"{name}: {stats['count']} items, Avg Vol: {avg_vol:.3f}, Top 5: {vols_list[:5]}")


if __name__ == "__main__":
    diag()
