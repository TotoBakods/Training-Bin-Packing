import math
import random
import numpy as np
import time
import multiprocessing
import gc
from functools import partial
from database import get_exclusion_zones

import atexit

# Global shared memory for multiprocessing (kept for backward compatibility)
_pool_items_props = None
_pool_wh_dims = None
_pool_valid_z = None
_pool_allocation_zones = None
_pool_exclusion_zones = None

_global_pool = None

def cleanup_pool():
    global _global_pool
    if _global_pool:
        _global_pool.terminate()
        _global_pool.join()
        _global_pool = None

atexit.register(cleanup_pool)

def get_process_pool():
    """Returns a singleton multiprocessing pool."""
    global _global_pool
    if _global_pool is None:
        cpu_count = multiprocessing.cpu_count()
        process_count = min(cpu_count, 12)
        _global_pool = multiprocessing.Pool(processes=process_count)
    return _global_pool

def init_worker(*args):
    """Deprecated: Initialization handled via explicit args now."""
    pass


def calculate_z_for_item(x, y, dim_x, dim_y, other_items_bbox, other_items_z, other_items_h, other_items_stackable=None, strict_stacking=True):
    """Calculate the lowest valid Z position for an item based on items below it."""
    if len(other_items_bbox) == 0:
        return 0.0
        
    # New item bounding box
    new_min_x = x - dim_x / 2
    new_max_x = x + dim_x / 2
    new_min_y = y - dim_y / 2
    new_max_y = y + dim_y / 2
    
    # Check XY plane overlaps (vectorized)
    overlaps_x = (new_min_x < other_items_bbox[:, 2]) & (new_max_x > other_items_bbox[:, 0])
    overlaps_y = (new_min_y < other_items_bbox[:, 3]) & (new_max_y > other_items_bbox[:, 1])
    overlaps = overlaps_x & overlaps_y
    
    if not np.any(overlaps):
        return 0.0
        
    # Get max Z top of overlapping items
    overlapping_z_tops = other_items_z[overlaps] + other_items_h[overlaps]

    # Reject stacking on non-stackable items
    if strict_stacking and other_items_stackable is not None:
        overlapping_stackables = other_items_stackable[overlaps]
        if np.any(overlapping_stackables == 0):
             return 1000000.0 # Effectively impossible
             
    max_z = np.max(overlapping_z_tops)
    
    # Stability check: ensure sufficient support area
    if max_z > 0:
        is_support = np.abs(overlapping_z_tops - max_z) < 0.01
        
        all_indices = np.arange(len(other_items_bbox))
        overlapping_indices = all_indices[overlaps]
        support_indices = overlapping_indices[is_support]
        
        # Calculate intersection area with supporting items
        sup_min_x = other_items_bbox[support_indices, 0]
        sup_min_y = other_items_bbox[support_indices, 1]
        sup_max_x = other_items_bbox[support_indices, 2]
        sup_max_y = other_items_bbox[support_indices, 3]
        
        inter_min_x = np.maximum(new_min_x, sup_min_x)
        inter_max_x = np.minimum(new_max_x, sup_max_x)
        inter_min_y = np.maximum(new_min_y, sup_min_y)
        inter_max_y = np.minimum(new_max_y, sup_max_y)
        
        w = np.maximum(0, inter_max_x - inter_min_x)
        h = np.maximum(0, inter_max_y - inter_min_y)
        area = w * h
        supported_area = np.sum(area)
        
        item_area = dim_x * dim_y
        if supported_area < (item_area * 0.2):  # 20% support threshold
            return max_z + 100000.0
            
    return max_z



def get_rotated_dims(l, w, h, rotation_code):
    """Returns (dx, dy, dz) based on rotation code 0-5."""
    code = int(rotation_code) % 6
    if code == 0: return l, w, h
    if code == 1: return w, l, h
    if code == 2: return l, h, w
    if code == 3: return h, l, w
    if code == 4: return w, h, l
    if code == 5: return h, w, l
    return l, w, h

def repair_solution_compact(solution, items_props=None, warehouse_dims=None, allocation_zones=None, layer_heights=None):
    """Repair solution by placing items in valid positions with gravity."""
    # Use globals if in worker process
    if items_props is None: items_props = _pool_items_props
    if warehouse_dims is None: warehouse_dims = _pool_wh_dims
    if allocation_zones is None: allocation_zones = _pool_allocation_zones
    
    # Defaults
    if layer_heights is None or len(layer_heights) == 0:
        layer_heights = [0.0]
    
    wh_len = warehouse_dims[0] if warehouse_dims else 100
    wh_wid = warehouse_dims[1] if warehouse_dims else 100
    wh_hgt = warehouse_dims[2] if warehouse_dims else 10
    
    num_items = len(solution)
    if num_items == 0: return solution

    # Sort indices:
    # Priority 1: Fragility (Ascending) - 0 (Robust) first, 1 (Fragile) last
    # Priority 2: Weight (Descending) - Heavy items first
    # Priority 3: Volume (Descending) - Large items first
    fragility = items_props[:, 8]
    weights = items_props[:, 6]
    volumes = items_props[:, 0] * items_props[:, 1] * items_props[:, 2]
    
    indices = np.arange(num_items)
    # Sort: fragile last, then heavy/large first
    sorted_indices = sorted(indices, key=lambda i: (fragility[i], -weights[i], -volumes[i], i))

    # Tracking placed items: (x, y, z, dx, dy, dz)
    placed_items = []
    
    # Use provided zones or default to full warehouse
    use_zones = allocation_zones if allocation_zones else [{'x1':0, 'y1':0, 'x2':wh_len, 'y2':wh_wid, 'z1':0, 'z2':wh_hgt}]

    for idx in sorted_indices:
        l, w, h = items_props[idx, 0:3]
        can_rotate = int(items_props[idx, 3])
        
        # Try flat rotations for stability
        rots = [0, 1] if can_rotate else [int(solution[idx, 3])]
        if can_rotate and items_props[idx, 4] == 1:
             pass  # Stick to flat rotations
        
        best_pos = None
        
        # Generate candidate positions
        candidates = set()
        
        # From zone corners
        for z in use_zones:
            candidates.add((z['x1'], z['y1']))
        
        # From placed items (adjacent positions)
        for (px, py, pz, pdx, pdy, pdz) in placed_items:
            candidates.add((px + pdx, py))  # Right
            candidates.add((px, py + pdy))  # Back
            candidates.add((px, py))        # On top
        
        # Add the gene's target position as a candidate to guide the heuristic
        # We add candidates for both rotated and unrotated orientations to be safe
        target_x = solution[idx, 0]
        target_y = solution[idx, 1]
        
        # Candidate 1: Assuming unrotated (l, w)
        cand1_x = target_x - items_props[idx, 0] / 2
        cand1_y = target_y - items_props[idx, 1] / 2
        candidates.add((cand1_x, cand1_y))
        
        # Candidate 2: Assuming rotated (w, l)
        cand2_x = target_x - items_props[idx, 1] / 2
        cand2_y = target_y - items_props[idx, 0] / 2
        candidates.add((cand2_x, cand2_y))

        
        # Filter valid candidates
        valid_candidates = []
        for (cx, cy) in candidates:
             if cx >= 0 and cy >= 0 and cx < wh_len and cy < wh_wid:
                 valid_candidates.append((cx, cy))
                 
        # Sort by proximity to optimizer's suggested position
        target_x = solution[idx, 0]
        target_y = solution[idx, 1]
        
        sorted_candidates = sorted(valid_candidates, key=lambda p: (
            (p[0] - target_x)**2 + (p[1] - target_y)**2,
            p[1], p[0]
        ))
        
        for rot in rots:
            dims = get_rotated_dims(l, w, h, rot)
            dx, dy, dz = dims
            
            for (cx, cy) in sorted_candidates:
                min_x, min_y = cx, cy
                max_x, max_y = cx + dx, cy + dy
                
                if max_x > wh_len + 0.001 or max_y > wh_wid + 0.001:
                    continue
                
                # Calculate gravity Z
                gravity_z = 0.0
                
                # Find highest item below this footprint
                for (px, py, pz, pdx, pdy, pdz) in placed_items:
                    if (max_x > px + 0.001 and min_x < px + pdx - 0.001 and
                        max_y > py + 0.001 and min_y < py + pdy - 0.001):
                        top_z = pz + pdz
                        if top_z > gravity_z:
                            gravity_z = top_z
                
                # Find valid Z in any suitable zone
                valid_z_found = False
                final_z = float('inf')
                
                for zne in use_zones:
                    # Check XY containment
                    if (min_x >= zne['x1'] - 0.01 and max_x <= zne['x2'] + 0.01 and 
                        min_y >= zne['y1'] - 0.01 and max_y <= zne['y2'] + 0.01):
                        
                        zone_floor = zne.get('z1', 0)
                        placement_z = max(gravity_z, zone_floor)
                        placement_top = placement_z + dz
                        zone_ceil = zne.get('z2', wh_hgt)
                        
                        # Check Z fits
                        if placement_top <= zone_ceil + 0.001:
                            if placement_z < final_z:
                                final_z = placement_z
                                valid_z_found = True
                    
                if valid_z_found:
                    # Calculate final score (Z, Dist to Target, Y, X)
                    # We prioritize Z (gravity) then closeness to gene target, then Y/X as tiebreaker
                    center_x = min_x + dx/2
                    center_y = min_y + dy/2
                    dist_to_target = (center_x - target_x)**2 + (center_y - target_y)**2
                    
                    score = (final_z, dist_to_target, min_y, min_x)
                    if best_pos is None or score < best_pos[7]:
                         best_pos = (center_x, center_y, final_z, rot, dx, dy, dz, score)
    
        # Apply placement
        if best_pos:
            b_x, b_y, b_z, b_rot, b_dx, b_dy, b_dz, _ = best_pos
        else:
            # Fallback: stack on top of everything
            b_z = 0
            if placed_items:
                max_top = max([p[2]+p[5] for p in placed_items])
                b_z = max_top
            
            b_rot = solution[idx, 3] if not can_rotate else 0
            dims = get_rotated_dims(l, w, h, b_rot)
            b_dx, b_dy, b_dz = dims
            b_x = dims[0]/2
            b_y = dims[1]/2
            
        solution[idx, 0] = b_x
        solution[idx, 1] = b_y
        solution[idx, 2] = b_z
        solution[idx, 3] = b_rot
        
        placed_items.append((b_x - b_dx/2, b_y - b_dy/2, b_z, b_dx, b_dy, b_dz))

    return solution


# Get valid Z positions for layers
def get_valid_z_positions(warehouse):
    if 'layer_heights' in warehouse and warehouse['layer_heights'] is not None:
        positions = set(warehouse['layer_heights'])
        positions.add(0.0)
        return sorted(list(positions))

    levels = warehouse.get('levels', 1)
    if levels <= 1:
        return [0.0]
    height = warehouse.get('height', 1)
    level_height = height / levels if levels > 0 else 0
    return [i * level_height for i in range(levels)]


# Standalone functions for multiprocessing

def create_random_solution_array(num_items, warehouse_dims=None, items_props=None, allocation_zones=None):
    """Create a random solution array with gravity-based placement."""
    # Use globals if running in worker
    if items_props is None: items_props = _pool_items_props
    if warehouse_dims is None: warehouse_dims = _pool_wh_dims
    if allocation_zones is None: allocation_zones = _pool_allocation_zones
    
    
    solution = np.zeros((num_items, 4), dtype=np.float32)
    wh_len, wh_wid, wh_hgt = warehouse_dims[:3]
    
    # Check for allocation zones
    has_allocation_zones = allocation_zones is not None and len(allocation_zones) > 0
    
    # Track placed items for gravity
    placed_bboxes = np.zeros((num_items, 4), dtype=np.float32)
    placed_z = np.zeros(num_items, dtype=np.float32)
    placed_h = np.zeros(num_items, dtype=np.float32)
    
    for i in range(num_items):
        item_len = items_props[i, 0]
        item_wid = items_props[i, 1]
        item_hgt = items_props[i, 2]
        can_rotate = items_props[i, 3]
        
        # Retry for floor priority (try to find Z=0)
        best_x, best_y, best_z = 0, 0, float('inf')
        best_rotation = 0
        
        for attempt in range(50):
            # Randomize rotation
            rotation = 0
            if can_rotate and random.random() > 0.5:
                    rotation = random.choice([0, 90, 180, 270])
            
            if int(rotation) % 180 == 0:
                dim_x, dim_y = item_len, item_wid
            else:
                dim_x, dim_y = item_wid, item_len
            
            # Position selection logic
            valid_zones = []
            if has_allocation_zones:
                for zone in allocation_zones:
                    zone_width = zone['x2'] - zone['x1']
                    zone_depth = zone['y2'] - zone['y1']
                    if dim_x <= zone_width and dim_y <= zone_depth:
                        valid_zones.append(zone)
            
            zone_z1 = 0
            zone_z2 = wh_hgt
            if valid_zones:
                # Sort zones by Z (bottom first)
                valid_zones.sort(key=lambda z: z.get('z1', 0))
                
                # Select zone sequentially
                zone_idx = attempt % len(valid_zones)
                zone_idx = min(zone_idx, len(valid_zones) - 1)
                zone = valid_zones[zone_idx]
                
                zone_z1 = zone.get('z1', 0)
                zone_z2 = zone.get('z2', wh_hgt)
                
                min_x = zone['x1'] + dim_x / 2
                max_x = zone['x2'] - dim_x / 2
                min_y = zone['y1'] + dim_y / 2
                max_y = zone['y2'] - dim_y / 2
                
                if max_x < min_x: max_x = min_x = (zone['x1'] + zone['x2']) / 2
                if max_y < min_y: max_y = min_y = (zone['y1'] + zone['y2']) / 2
                
                # Dense packing: try corner first, then adjacent, then random
                if attempt < 5:
                    x = min_x
                    y = min_y
                elif attempt < 45 and i > 0:
                    # Place adjacent to existing item
                    rand_idx = random.randint(0, i-1)
                    ref_box = placed_bboxes[rand_idx]
                    if random.random() < 0.5:
                        x = ref_box[2] + dim_x / 2
                        y = ref_box[1] + dim_y / 2
                    else:
                        x = ref_box[0] + dim_x / 2
                        y = ref_box[3] + dim_y / 2
                    x += random.uniform(-1, 1)
                    y += random.uniform(-1, 1)
                    x = max(min_x, min(max_x, x))
                    y = max(min_y, min(max_y, y))
                else:
                    x = random.uniform(min_x, max_x)
                    y = random.uniform(min_y, max_y)
                    
            else:
                min_x = dim_x / 2
                max_x = wh_len - dim_x / 2
                min_y = dim_y / 2
                max_y = wh_wid - dim_y / 2
                
                if max_x < min_x: max_x = min_x
                if max_y < min_y: max_y = min_y
                
            # Random placement logic - removed deterministic corner bias to ensure population diversity
            # Use strict random uniform placement for all attempts to prevent clones
            if has_allocation_zones:
                 # Zone logic... similar randomization needed
                 # For brevity, let's assume global logic first or fix zone logic too
                 # Actually, logic below handles both.
                 # Just use random.uniform for the coordinates.
                 pass

            # Simplified Random Logic (Global & Zone)
            # We already calculated min_x, max_x etc in previous lines
            # Just use them.
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            
            # Check Z immediately
            z = calculate_z_for_item(x, y, dim_x, dim_y, placed_bboxes[:i], placed_z[:i], placed_h[:i])
            
            # Enforce layer floor
            z = max(z, zone_z1)
            
            # Snap to next layer if exceeds ceiling
            if z + item_hgt > zone_z2 and zone_z2 < wh_hgt:
                z = zone_z2
            
            if z < best_z:
                best_x, best_y, best_z, best_rotation = x, y, z, rotation
            
            if z == zone_z1:
                break
        
        if best_z > 50000:
            best_z -= 100000.0
        
        x, y, z = best_x, best_y, best_z
        rotation = best_rotation
        
        # Recalculate dims for best rotation
        if int(rotation) % 180 == 0:
            dim_x, dim_y = item_len, item_wid
        else:
            dim_x, dim_y = item_wid, item_len
        
        # Store
        solution[i] = [x, y, z, rotation]
        
        # Update tracking arrays for next items
        placed_bboxes[i] = [x - dim_x/2, y - dim_y/2, x + dim_x/2, y + dim_y/2]
        placed_z[i] = z
        placed_h[i] = item_hgt
        
    return solution

def fitness_function_numpy(solution, items_props=None, warehouse_dims=None, weights=None, valid_z=None, exclusion_zones_arr=None):
    # Use globals if in worker process
    if items_props is None: items_props = _pool_items_props
    if warehouse_dims is None: warehouse_dims = _pool_wh_dims
    if valid_z is None: valid_z = _pool_valid_z
    if exclusion_zones_arr is None: exclusion_zones_arr = _pool_exclusion_zones
    
    # solution: (N, 4)
    # items_props: (N, 8) cols: len, wid, hgt, can_rot, stackable, access_freq, weight, category_id
    # exclusion_zones_arr: (K, 4) -> x1, y1, x2, y2
    
    # Ensure float32 for memory efficiency
    solution = solution.astype(np.float32, copy=False)
    # items_props is likely already float32 if we initialized it carefully, but let's assume it's ro (read-only)
    
    # Calculate Space Utilization
    # Calculate Space Utilization
    grouping = 0.0 # Initialize early to avoid NameError
    total_vol = np.sum(items_props[:, 0] * items_props[:, 1] * items_props[:, 2])
    wh_vol = warehouse_dims[0] * warehouse_dims[1] * warehouse_dims[2]
    space_util = total_vol / wh_vol if wh_vol > 0 else 0
    
    # Calculate Accessibility (Distance to door)
    door_x, door_y = 0, 0
    if len(warehouse_dims) >= 5:
        door_x, door_y = warehouse_dims[3], warehouse_dims[4]
        
    dists = np.sqrt((solution[:, 0] - door_x)**2 + (solution[:, 1] - door_y)**2)
    # Avoid div by zero
    access_scores = 1.0 / (1.0 + dists)
    
    freqs = items_props[:, 5]
    if np.sum(freqs) > 1e-9:
        accessibility = np.average(access_scores, weights=freqs)
    else:
        accessibility = np.mean(access_scores)
    
    # Stability (Geometric Support)
    # Check if each item is supported by the floor (z=0) or by another item below it.
    
    # Pre-calculate rotated dimensions for all items
    # Rotations: 0=LWH, 1=WLH, 2=LHW, 3=HLW, 4=WHL, 5=HWL
    rots = solution[:, 3].astype(int)
    l_arr = items_props[:, 0]
    w_arr = items_props[:, 1]
    h_arr = items_props[:, 2] # Original scalar height
    
    # We need CURRENT dimensions based on rotation
    cur_l = np.zeros(len(solution))
    cur_w = np.zeros(len(solution))
    cur_h = np.zeros(len(solution))
    
    # Vectorized dimension calculation
    for r_code in range(6):
        mask = (rots % 6 == r_code)
        if not np.any(mask): continue
        if r_code == 0: cur_l[mask], cur_w[mask], cur_h[mask] = l_arr[mask], w_arr[mask], h_arr[mask]
        elif r_code == 1: cur_l[mask], cur_w[mask], cur_h[mask] = w_arr[mask], l_arr[mask], h_arr[mask]
        elif r_code == 2: cur_l[mask], cur_w[mask], cur_h[mask] = l_arr[mask], h_arr[mask], w_arr[mask]
        elif r_code == 3: cur_l[mask], cur_w[mask], cur_h[mask] = h_arr[mask], l_arr[mask], w_arr[mask]
        elif r_code == 4: cur_l[mask], cur_w[mask], cur_h[mask] = w_arr[mask], h_arr[mask], l_arr[mask]
        elif r_code == 5: cur_l[mask], cur_w[mask], cur_h[mask] = h_arr[mask], w_arr[mask], l_arr[mask]

    is_stable = np.zeros(len(solution), dtype=bool)
    
    # 1. Floor Support
    is_stable |= (solution[:, 2] <= 0.01)
    
    # 2. Item-on-Item Support (O(N^2) naive, but fast enough for <1000 items in numpy)
    # Ideally use a spatial grid, but let's try vectorized broadcasting or loop if N is small
    # For robust "Tetris", items must be supported.
    
    # Optimization: items are sorted by Z usually? No.
    # Let's iterate. Item i is supported if exists j s.t. j is below i and overlaps horizontally
    
    # Extract coordinates (N,)
    x = solution[:, 0]
    y = solution[:, 1]
    z = solution[:, 2]
    
    # For every unstable item, check if there is a supporter
    unstable_indices = np.where(~is_stable)[0]
    
    if len(unstable_indices) > 0:
        # Check support for unstable items
        # Support condition:
        # 1. j top is near i bottom: abs(z[j] + h[j] - z[i]) < tolerance
        # 2. Horizontal overlap > 0 (or > threshold)
        
        # We can loop through unstable items and vector-check against all others
        for i in unstable_indices:
            # Candidates: items strictly below i
            candidates = np.where(z + cur_h < z[i] + 0.05)[0] 
            if len(candidates) == 0: continue
            
            # Filter matches: z_top_j ~ z_bottom_i
            z_diff = np.abs((z[candidates] + cur_h[candidates]) - z[i])
            z_match = z_diff < 0.05
            vertical_supporters = candidates[z_match]
            
            if len(vertical_supporters) == 0: continue
            
            # Check horizontal overlap
            # Item i bounds
            ix1, ix2 = x[i] - cur_l[i]/2, x[i] + cur_l[i]/2
            iy1, iy2 = y[i] - cur_w[i]/2, y[i] + cur_w[i]/2
            
            # Supporter bounds
            sx1 = x[vertical_supporters] - cur_l[vertical_supporters]/2
            sx2 = x[vertical_supporters] + cur_l[vertical_supporters]/2
            sy1 = y[vertical_supporters] - cur_w[vertical_supporters]/2
            sy2 = y[vertical_supporters] + cur_w[vertical_supporters]/2
            
            # Overlap logic
            # overlap_x = max(0, min(ix2, sx2) - max(ix1, sx1))
            ox = np.maximum(0, np.minimum(ix2, sx2) - np.maximum(ix1, sx1))
            oy = np.maximum(0, np.minimum(iy2, sy2) - np.maximum(iy1, sy1))
            
            area = ox * oy
            # Configurable support threshold (e.g., 30% area or just > 0)
            # For strictly stable packing, usually > 50% or center of mass support.
            # Simplified: if total overlapping area provides enough support?
            # Or simpler: if ANY support > small_area
            if np.any(area > (cur_l[i] * cur_w[i]) * 0.2): # 20% support
                is_stable[i] = True

    stability = np.mean(is_stable)
    
    # Exclusion Zones
    zone_penalty = 0
    if exclusion_zones_arr is not None and len(exclusion_zones_arr) > 0:
        x = solution[:, 0:1] # (N, 1)
        y = solution[:, 1:2] # (N, 1)
        
        z_x1 = exclusion_zones_arr[:, 0] # (K,)
        z_y1 = exclusion_zones_arr[:, 1]
        z_x2 = exclusion_zones_arr[:, 2]
        z_y2 = exclusion_zones_arr[:, 3]
        
        # Better: AABB overlap
        # Item dims (approximation with non-rotated len/wid for speed or use max dim)
        radii = np.maximum(items_props[:, 0], items_props[:, 1]) / 2.0
        radii = radii.reshape(-1, 1)
        
        # Zone centers/dims
        z_cx = (z_x1 + z_x2) / 2
        z_cy = (z_y1 + z_y2) / 2
        z_hw = (z_x2 - z_x1) / 2
        z_hh = (z_y2 - z_y1) / 2
        
        # Distance from center to center per axis
        dx = np.abs(x - z_cx)
        dy = np.abs(y - z_cy)
        
        # Vectorized AABB with rotation ignored (using max dim covers worst case)
        collision_x = dx < (radii + z_hw)
        collision_y = dy < (radii + z_hh)
        collisions = collision_x & collision_y
        
        zone_penalty = np.sum(collisions) / len(solution) # Fraction of items colliding
        
    # --- Item-Item Overlap ---
    # Optimized to use batches to avoid O(N^2) memory usage for large N.
    n = len(solution)
    overlap_count = 0
    if n > 0:
        # Extract centers (n, 1)
        x = solution[:, 0]
        y = solution[:, 1]
        z = solution[:, 2]  # Bottom
        h = items_props[:, 2]  # Heights
        
        # Calculate actual dimensions based on rotation
        rots = solution[:, 3].astype(int)
        l = items_props[:, 0]
        w = items_props[:, 1]
        orig_h = items_props[:, 2] # Need original height to swap
        
        # We need to vectorizely apply get_rotated_dims?
        # get_rotated_dims is not vectorized.
        # But we can simulate it with numpy usage.
        
        # Codes 0-5
        # 0: L, W, H
        # 1: W, L, H
        # 2: L, H, W
        # 3: H, L, W
        # 4: W, H, L
        # 5: H, W, L
        
        current_len = np.zeros(n, dtype=np.float32)
        current_wid = np.zeros(n, dtype=np.float32)
        current_hgt = np.zeros(n, dtype=np.float32)
        
        rot_mod = rots % 6
        
        # Case 0
        mask = (rot_mod == 0)
        current_len[mask] = l[mask]
        current_wid[mask] = w[mask]
        current_hgt[mask] = orig_h[mask]
        
        # Case 1
        mask = (rot_mod == 1)
        current_len[mask] = w[mask]
        current_wid[mask] = l[mask]
        current_hgt[mask] = orig_h[mask]
        
        # Case 2
        mask = (rot_mod == 2)
        current_len[mask] = l[mask]
        current_wid[mask] = orig_h[mask]
        current_hgt[mask] = w[mask]
        
        # Case 3
        mask = (rot_mod == 3)
        current_len[mask] = orig_h[mask]
        current_wid[mask] = l[mask]
        current_hgt[mask] = w[mask]
        
        # Case 4
        mask = (rot_mod == 4)
        current_len[mask] = w[mask]
        current_wid[mask] = orig_h[mask]
        current_hgt[mask] = l[mask]
        
        # Case 5
        mask = (rot_mod == 5)
        current_len[mask] = orig_h[mask]
        current_wid[mask] = w[mask]
        current_hgt[mask] = l[mask]
        
        # Half-dims
        hw = current_len / 2
        hh = current_wid / 2
        
        # Z intervals (bottom + rotated height)
        z1 = z
        z2 = z + current_hgt
        
        # Reduced Batch Size for Memory Safety
        BATCH_SIZE = 512 
        
        for i_start in range(0, n, BATCH_SIZE):
            i_end = min(i_start + BATCH_SIZE, n)
            
            # Batch slices
            x_batch = x[i_start:i_end].reshape(-1, 1)  # (B, 1)
            y_batch = y[i_start:i_end].reshape(-1, 1)
            z1_batch = z1[i_start:i_end].reshape(-1, 1)
            z2_batch = z2[i_start:i_end].reshape(-1, 1)
            hw_batch = hw[i_start:i_end].reshape(-1, 1)
            hh_batch = hh[i_start:i_end].reshape(-1, 1)
            
            # Inner loop batching to keep memory low
            for j_start in range(0, n, BATCH_SIZE):
                j_end = min(j_start + BATCH_SIZE, n)
                
                # Check bounds to skip duplicate work if we wanted to triangularize, 
                # but calculating full matrix blockwise is simpler to vectorize.
                
                x_other = x[j_start:j_end].reshape(1, -1)  # (1, B2)
                y_other = y[j_start:j_end].reshape(1, -1)
                z1_other = z1[j_start:j_end].reshape(1, -1)
                z2_other = z2[j_start:j_end].reshape(1, -1)
                hw_other = hw[j_start:j_end].reshape(1, -1)
                hh_other = hh[j_start:j_end].reshape(1, -1)
                
                # Overlap checks
                # X overlap: |x1 - x2| < hw1 + hw2
                dx = np.abs(x_batch - x_other)
                overlap_x = dx < (hw_batch + hw_other - 0.01) # 1cm tolerance? No, stricter.
                
                # Y overlap
                dy = np.abs(y_batch - y_other)
                overlap_y = dy < (hh_batch + hh_other - 0.01)
                
                # Z overlap
                # Interval overlap: not (end1 <= start2 or start1 >= end2)
                # Strict < inequality implies 0 thickness overlap is ignored, which is good.
                overlap_z = (z2_batch > (z1_other + 0.01)) & (z1_batch < (z2_other - 0.01))
                
                # Combined
                overlaps = overlap_x & overlap_y & overlap_z
                
                overlap_count += np.sum(overlaps)
        
        # Remove self-overlaps (diagonal was counted once per item)
        # Each item overlaps with itself in the logic above.
        overlap_count -= n
        
        # Divide by 2 because A-B and B-A are counted
        overlap_count /= 2.0
        
        # Draconian Penalty
        # Soft Normalized Penalty
        if overlap_count > 0:
             overlap_penalty = overlap_count / n # Overlaps per item average
        else:
             overlap_penalty = 0.0
        
    # --- Stackability Enforcement ---
    # Check if items are stacked on non-stackable items
    stackability_penalty = 0
    
    # Optimized Vectorized Stackability Check
    # n is already defined
    if n > 1:
        # Reuse variables extracted earlier
        x = solution[:, 0]
        y = solution[:, 1]
        z = solution[:, 2]
        h = items_props[:, 2]  # heights
        stackable = items_props[:, 4]  # stackable flags
        
        # Get item footprint dimensions (accounting for rotation)
        # Get item footprint dimensions (accounting for rotation)
        rots = solution[:, 3].astype(int)
        l = items_props[:, 0]
        w = items_props[:, 1]
        orig_h = items_props[:, 2]
        
        # 6-Axis Dimension Logic (Vectorized)
        current_len = np.zeros(n, dtype=np.float32)
        current_wid = np.zeros(n, dtype=np.float32)
        current_hgt = np.zeros(n, dtype=np.float32) # We need this for z_tops!
        
        rot_mod = rots % 6
        
        # Case 0 (L, W, H)
        mask = (rot_mod == 0)
        current_len[mask] = l[mask]; current_wid[mask] = w[mask]; current_hgt[mask] = orig_h[mask]
        # Case 1 (W, L, H)
        mask = (rot_mod == 1)
        current_len[mask] = w[mask]; current_wid[mask] = l[mask]; current_hgt[mask] = orig_h[mask]
        # Case 2 (L, H, W)
        mask = (rot_mod == 2)
        current_len[mask] = l[mask]; current_wid[mask] = orig_h[mask]; current_hgt[mask] = w[mask]
        # Case 3 (H, L, W)
        mask = (rot_mod == 3)
        current_len[mask] = orig_h[mask]; current_wid[mask] = l[mask]; current_hgt[mask] = w[mask]
        # Case 4 (W, H, L)
        mask = (rot_mod == 4)
        current_len[mask] = w[mask]; current_wid[mask] = orig_h[mask]; current_hgt[mask] = l[mask]
        # Case 5 (H, W, L)
        mask = (rot_mod == 5)
        current_len[mask] = orig_h[mask]; current_wid[mask] = w[mask]; current_hgt[mask] = l[mask]

        hw = current_len / 2
        hh = current_wid / 2
        
        # Z-tops MUST use rotated height
        z_tops = z + current_hgt
        
        # We need to find pairs (i, j) where i is resting on j.
        # Resting condition: abs(z[i] - z_top[j]) < 0.1
        # AND Footprint Overlap
        
        violations = 0
        BATCH_SIZE_STACK = 128 # Small batch for safety
        
        for i_start in range(0, n, BATCH_SIZE_STACK):
            i_end = min(i_start + BATCH_SIZE_STACK, n)
            
            # Batch I data (Potential Top Items)
            z_i = z[i_start:i_end].reshape(-1, 1) # (B, 1)
            x_i = x[i_start:i_end].reshape(-1, 1)
            y_i = y[i_start:i_end].reshape(-1, 1)
            hw_i = hw[i_start:i_end].reshape(-1, 1)
            hh_i = hh[i_start:i_end].reshape(-1, 1)
            
            # Filter: Only check items that are NOT on the ground
            # effective_mask = (z_i > 0.01).flatten()
            # If we want to optimize further we could skip ground items, but vectorization is fast enough.
            
            for j_start in range(0, n, BATCH_SIZE_STACK):
                j_end = min(j_start + BATCH_SIZE_STACK, n)
                
                # Batch J data (Potential Support Items)
                z_j_top = z_tops[j_start:j_end].reshape(1, -1) # (1, B2)
                
                # Z-Check: Is i resting on j?
                # resting = abs(z_i - z_j_top) < 0.1
                resting = np.abs(z_i - z_j_top) < 0.1
                
                if not np.any(resting):
                    continue
                    
                # XY Overlap Check for resting pairs
                x_j = x[j_start:j_end].reshape(1, -1)
                y_j = y[j_start:j_end].reshape(1, -1)
                hw_j = hw[j_start:j_end].reshape(1, -1)
                hh_j = hh[j_start:j_end].reshape(1, -1)
                
                dx = np.abs(x_i - x_j)
                dy = np.abs(y_i - y_j)
                
                # Overlap Threshold (50% rule mostly... logic was: < (hw1+hw2)*0.5)
                # Wait, original logic: overlap_threshold_x = (hw[i] + hw[j]) * 0.5
                # This means centers must be VERY close.
                # Actually (hw[i] + hw[j]) is the touching distance. * 0.5 means they must overlap by 50%?
                # Yes.
                
                thresh_x = (hw_i + hw_j) * 0.5
                thresh_y = (hh_i + hh_j) * 0.5
                
                xy_overlap = (dx < thresh_x) & (dy < thresh_y)
                
                # Valid Support Relation
                is_supported = resting & xy_overlap
                
                # Self-support is impossible due to z vs z+h check (unless h=0, but valid check prevents self-loop effectively)
                # But to be safe, if i==j, z_i cannot equal z_j + h_j unless h_j=0.
                
                if np.any(is_supported):
                     # Check if supporter J is stackable
                     stackable_j = stackable[j_start:j_end].reshape(1, -1) # (1, B2)
                     
                     # Identify bad supports: Supported by item with stackable=0
                     # mask: is_supported AND (stackable_j == 0)
                     bad_support = is_supported & (stackable_j < 0.5)
                     
                     # Count unique ITEMS 'i' that have at least one bad support
                     # Reduce along J axis: does item i have ANY bad support in this batch?
                     has_bad_support = np.any(bad_support, axis=1) # (B,)
                     
                     violations += np.sum(has_bad_support)

        stackability_penalty = violations / n if n > 0 else 0
    
    # --- Grouping Metric ---
    grouping = 0.0
    n_items = len(solution)
    if n_items > 0:
        cats = items_props[:, 7] # Category hash
        unique_cats = np.unique(cats)
        
        total_dist_sum = 0
        count = 0
        
        x = solution[:, 0]
        y = solution[:, 1]
        
        for cat in unique_cats:
            # Mask for this category
            mask = (cats == cat)
            if np.sum(mask) <= 1:
                continue # Single item already grouped with itself
                
            # Centroid
            c_x = np.mean(x[mask])
            c_y = np.mean(y[mask])
            
            # Distances to centroid
            dists = np.sqrt((x[mask] - c_x)**2 + (y[mask] - c_y)**2)
            
            total_dist_sum += np.sum(dists)
            count += np.sum(mask)
        
        if count > 0:
            avg_dist = total_dist_sum / count
            grouping = 1.0 / (1.0 + avg_dist * 0.1)
        else:
            grouping = 1.0 # Perfect grouping if all singles or empty
    else:
            grouping = 0

    
    total_weight = sum(weights.values())
    if total_weight <= 1e-9:
        norm_weights = weights # Avoid division by zero
    else:
        norm_weights = {k: v / total_weight for k, v in weights.items()}
    
    # Penalize fitness for zone violations
    fitness = (norm_weights.get('space', 0) * space_util +
                norm_weights.get('accessibility', 0) * accessibility +
                norm_weights.get('stability', 0) * stability +
                norm_weights.get('grouping', 0) * grouping)
    
    if zone_penalty > 0:
        fitness *= (1.0 / (1.0 + zone_penalty * 5.0)) # Smooth gradient

    if overlap_penalty > 0:
        fitness *= (1.0 / (1.0 + overlap_penalty * 5.0)) # Smooth gradient

    # Apply stackability penalty - items on non-stackable items
    if stackability_penalty > 0:
        fitness *= (1.0 / (1.0 + stackability_penalty * 5.0))

    if fitness <= 1e-6:
        # Debug only randomly to save IO
        if random.random() < 0.001:
            with open('thread_debug.log', 'a') as f:
                f.write(f"Zero Fit: Overlap={overlap_penalty:.4f}, Zone={zone_penalty:.4f}, Stack={stackability_penalty:.4f}\n")

    # Prefer lower Z (floor usage) if possible - reducing fitness slightly as average Z increases
    avg_z = np.mean(solution[:, 2])
    wh_hgt_val = warehouse_dims[2]
    if wh_hgt_val > 0:
        fitness *= (1.0 - (avg_z / wh_hgt_val) * 0.15) 

    return fitness, space_util, accessibility, stability, grouping

def create_and_repair(num_items, warehouse_dims, items_props, allocation_zones, valid_z):
    # Robust seeding using OS entropy, time, and PID to ensure diversity
    import os, time, threading
    seed_val = (int(time.time() * 1000000) ^ os.getpid() ^ threading.get_ident() ^ random.randint(0, 1000000)) & 0xFFFFFFFF
    random.seed(seed_val)
    np.random.seed(seed_val)
    
    sol = create_random_solution_array(num_items, warehouse_dims, items_props, allocation_zones)
    sol = repair_solution_compact(sol, items_props, warehouse_dims, allocation_zones, valid_z)
    return sol


def fitness_function(solution_list, items, warehouse, weights=None):
    # Convert list of dicts back to numpy for calc
    import numpy as np
    num_items = len(items)
    # Create map
    item_map = {item['id']: i for i, item in enumerate(items)}
    
    sol_array = np.zeros((num_items, 4))
    
    for item_sol in solution_list:
        idx = item_map.get(item_sol['id'])
        if idx is not None:
            sol_array[idx] = [item_sol['x'], item_sol['y'], item_sol['z'], item_sol['rotation']]
            
    items_props = np.zeros((num_items, 9))
    for i, item in enumerate(items):
        items_props[i] = [
            item['length'], item['width'], item['height'],
            item['can_rotate'], item['stackable'],
            item['access_freq'], item.get('weight', 0),
            hash(item.get('category', '')) % 10000,
            item.get('fragility', 0)
        ]
        
    wh_dims = (warehouse['length'], warehouse['width'], warehouse['height'], 
               warehouse.get('door_x', 0), warehouse.get('door_y', 0))
    valid_z = get_valid_z_positions(warehouse)
    
    from database import get_exclusion_zones
    zones = get_exclusion_zones(warehouse['id'])
    exclusion_zones_arr = None
    if zones:
         ex_zones = [z for z in zones if z['zone_type'] == 'exclusion']
         if ex_zones:
             exclusion_zones_arr = np.array([[z['x1'], z['y1'], z['x2'], z['y2']] for z in ex_zones])

    return fitness_function_numpy(sol_array, items_props, wh_dims, 
        weights or {'space': 0.5}, valid_z, exclusion_zones_arr)

def calculate_center_of_gravity(solution_list, items_dict):
    total_mass = 0
    mx, my, mz = 0, 0, 0
    for sol in solution_list:
        item = items_dict.get(sol['id'])
        if item:
            mass = item.get('weight', 0)
            mx += mass * sol['x']
            my += mass * sol['y']
            mz += mass * (sol['z'] + item['height']/2)
            total_mass += mass
    if total_mass == 0: return 0,0,0
    return mx/total_mass, my/total_mass, mz/total_mass
