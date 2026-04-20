import math
import random
import numpy as np
import multiprocessing
from database import get_exclusion_zones
import os
import datetime
print(f'>>> OPTIMIZER MODULE LOADED FROM: {__file__}')

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
    _TORCH_DEVICE = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
except ImportError:
    _TORCH_AVAILABLE = False
    _TORCH_DEVICE = None

# Global variables for multiprocessing pool
_pool_items_props = None
_pool_wh_dims = None
_pool_valid_z = None
_pool_exclusion_zones = None


class SimpleGrid:
    """A 2D Spatial Grid to speed up item-item overlap and gravity checks."""
    def __init__(self, wh_l, wh_w, cell_size=2.0):
        self.cell_size = cell_size
        self.cols = max(1, math.ceil(wh_l / cell_size))
        self.rows = max(1, math.ceil(wh_w / cell_size))
        # Grid of sets containing indices of placed items
        self.grid = [ [set() for _ in range(self.rows)] for _ in range(self.cols) ]

    def _get_cells(self, x1, y1, x2, y2):
        c1 = max(0, min(self.cols-1, int(x1 / self.cell_size)))
        c2 = max(0, min(self.cols-1, int(x2 / self.cell_size)))
        r1 = max(0, min(self.rows-1, int(y1 / self.cell_size)))
        r2 = max(0, min(self.rows-1, int(y2 / self.cell_size)))
        return c1, c2, r1, r2

    def insert(self, idx, x1, y1, x2, y2):
        c1, c2, r1, r2 = self._get_cells(x1, y1, x2, y2)
        for c in range(c1, c2 + 1):
            for r in range(r1, r2 + 1):
                self.grid[c][r].add(idx)

    def query(self, x1, y1, x2, y2):
        c1, c2, r1, r2 = self._get_cells(x1, y1, x2, y2)
        matches = set()
        for c in range(c1, c2 + 1):
            for r in range(r1, r2 + 1):
                matches.update(self.grid[c][r])
        if not matches: return np.array([], dtype=int)
        return np.array(list(matches), dtype=int)




def calculate_z_for_item(x, y, dim_x, dim_y, other_items_bbox, other_items_z, other_items_h, other_items_stackable=None, strict_stacking=True, grid=None):
    """Calculate the lowest valid Z position for an item based on items below it."""
    # New item bounding box
    new_min_x = x - dim_x / 2
    new_max_x = x + dim_x / 2
    new_min_y = y - dim_y / 2
    new_max_y = y + dim_y / 2

    # Spatial Query Optimization
    if grid is not None:
        relevant_indices = grid.query(new_min_x, new_min_y, new_max_x, new_max_y)
        if len(relevant_indices) == 0:
            return 0.0
        # Subset relevant items
        other_items_bbox = other_items_bbox[relevant_indices]
        other_items_z = other_items_z[relevant_indices]
        other_items_h = other_items_h[relevant_indices]
        if other_items_stackable is not None:
            other_items_stackable = other_items_stackable[relevant_indices]

    if len(other_items_bbox) == 0:
        return 0.0
        
    # NEW: Filter overlapping items to only those that can actually support this item.
    # We ignore items whose base (z) is higher than the top of any potential 
    # support we've already found + some reasonable cushion, but more importantly,
    # in multi-zone setups, we should only care about items that are actually
    # below or at the level we are trying to place.
    
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
        if supported_area < (item_area * 0.12):  # 12% support threshold — allows tighter stacking
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

def _search_candidates_gpu(cands_xy, rots_list, l, w, h,
                            placed_buf, placed_count,
                            use_zones, assigned_zone,
                            wh_len, wh_wid, wh_hgt,
                            target_x, target_y, is_fast, device,
                            min_z=0.0,
                            door_x=0.0, door_y=0.0, af=0.0, prio=1.0,
                            zx1=0.0, zy1=0.0, zx2=0.0, zy2=0.0):
    """Vectorized (GPU or CPU-torch) candidate search with rotation-aware tight packing.
    For each rotation, appends flush-left/front/right-align/back-align candidates
    derived from placed items so every tight-fit position is evaluated.
    Returns (center_x, center_y, z, rot, dx, dy, dz, score) or None."""
    import torch

    has_base = len(cands_xy) > 0
    has_placed = placed_count > 0

    if not has_base and not has_placed:
        return None

    if has_base:
        base_t  = torch.tensor(cands_xy, dtype=torch.float32, device=device)
        base_cx = base_t[:, 0]
        base_cy = base_t[:, 1]

    if has_placed:
        pd         = placed_buf[:placed_count]    # (P, 6)
        z_tops_all = pd[:, 2] + pd[:, 5]          # (P,)
        valid_sup  = pd[:, 2] < 1000              # (P,)
        px_p  = pd[:, 0];  py_p  = pd[:, 1]
        pdx_p = pd[:, 3];  pdy_p = pd[:, 4]

    # Restrict zone validation to the item's assigned zone so items can't
    # drift into a neighbouring zone and disrupt the fragility layering.
    valid_zones = [assigned_zone] if assigned_zone is not None else use_zones
    # Pack from the zone's front-left corner for compactness.
    # Falls back to warehouse origin (0,0) when no assignment is active.
    zone_ox = float(assigned_zone['x1']) if assigned_zone is not None else 0.0
    zone_oy = float(assigned_zone['y1']) if assigned_zone is not None else 0.0
    # Bounds of the assigned zone used to clip tight candidates (None = no clip)
    az_x1 = float(assigned_zone['x1']) if assigned_zone is not None else None
    az_x2 = float(assigned_zone['x2']) if assigned_zone is not None else None
    az_y1 = float(assigned_zone['y1']) if assigned_zone is not None else None
    az_y2 = float(assigned_zone['y2']) if assigned_zone is not None else None

    BIG  = torch.tensor(1e9,  dtype=torch.float32, device=device)
    best = None

    for rot_code in rots_list:
        dx, dy, dz = get_rotated_dims(l, w, h, rot_code)
        dx_f, dy_f, dz_f = float(dx), float(dy), float(dz)

        # --- Build per-rotation candidate list ---
        # Base (lattice + adjacency from outer loop) + 6 tight positions per placed item
        if has_placed:
            tc_x = torch.cat([
                px_p + pdx_p,           # flush right:   new item left  at placed right
                px_p,                   # flush back:    same X
                px_p - dx_f,            # flush left:    new item right at placed left  (rotation-aware)
                px_p,                   # flush front:   same X
                px_p + pdx_p - dx_f,    # right-align:   new item right at placed right (rotation-aware)
                px_p,                   # back-align:    same X
            ])
            tc_y = torch.cat([
                py_p,                   # flush right:   same Y
                py_p + pdy_p,           # flush back:    new item front at placed back
                py_p,                   # flush left:    same Y
                py_p - dy_f,            # flush front:   new item back  at placed front (rotation-aware)
                py_p,                   # right-align:   same Y
                py_p + pdy_p - dy_f,    # back-align:    new item back  at placed back  (rotation-aware)
            ])
            # Clip tight candidates to assigned zone so they don't
            # pull items into a neighbouring zone.
            if az_x1 is not None:
                tight_ok = (tc_x >= az_x1 - 0.01) & (tc_x + dx_f <= az_x2 + 0.01) & \
                           (tc_y >= az_y1 - 0.01) & (tc_y + dy_f <= az_y2 + 0.01)
                if tight_ok.any():
                    tc_x, tc_y = tc_x[tight_ok], tc_y[tight_ok]
                else:
                    tc_x = tc_x[:0]; tc_y = tc_y[:0]  # empty

            all_cx = torch.cat([base_cx, tc_x]) if has_base else tc_x
            all_cy = torch.cat([base_cy, tc_y]) if has_base else tc_y
        else:
            if not has_base:
                continue
            all_cx, all_cy = base_cx, base_cy

        max_x_t = all_cx + dx_f
        max_y_t = all_cy + dy_f
        N_t = all_cx.shape[0]

        # Bounds check (includes cx/cy >= 0 via wall-clipping)
        valid_bounds = (all_cx >= 0) & (all_cy >= 0) & \
                       (max_x_t <= wh_len + 0.001) & (max_y_t <= wh_wid + 0.001)
        if not valid_bounds.any():
            continue

        # --- Gravity Z: batch AABB overlap (N_t × P) ---
        if has_placed:
            cx_e  = all_cx.unsqueeze(1);   cy_e  = all_cy.unsqueeze(1)
            mx_e  = max_x_t.unsqueeze(1);  my_e  = max_y_t.unsqueeze(1)
            px_e  = px_p.unsqueeze(0);     py_e  = py_p.unsqueeze(0)
            pdx_e = pdx_p.unsqueeze(0);    pdy_e = pdy_p.unsqueeze(0)

            # Strict AABB overlap for gravity: items overlapping by > 0 are detected.
            # Using 1e-6 epsilon for float32 precision safety only — not a physics gap.
            _OVL = 1e-6
            ov_x = (mx_e > px_e + _OVL) & (cx_e < px_e + pdx_e - _OVL)
            ov_y = (my_e > py_e + _OVL) & (cy_e < py_e + pdy_e - _OVL)
            
            # Multi-zone independence: items already in zones ABOVE our ceiling
            # must not affect gravity/stacking in the current zone.
            # If we are searching for assigned_zone, only items with base Z 
            # below that zone's ceiling are valid.
            if assigned_zone is not None:
                az_z2 = float(assigned_zone.get('z2', wh_hgt))
                valid_for_gravity = valid_sup & (pd[:, 2] < az_z2 - 0.001)
            else:
                valid_for_gravity = valid_sup

            overlaps = ov_x & ov_y & valid_for_gravity.unsqueeze(0)          # (N_t, P)

            z_tops_e = z_tops_all.unsqueeze(0).expand(N_t, -1)
            gz = torch.where(overlaps, z_tops_e, torch.zeros_like(z_tops_e))
            gravity_z_t = gz.max(dim=1).values                        # (N_t,)
            
            # --- Physical Stability Check ---
            # An item is stable if it rests on the floor OR if the items directly beneath it
            # provide a combined supporting surface area of at least 50% of the candidate's base area.
            is_support = overlaps & (z_tops_e >= gravity_z_t.unsqueeze(1) - 1e-4)
            
            ix_min = torch.max(cx_e, px_e)
            ix_max = torch.min(mx_e, px_e + pdx_e)
            iy_min = torch.max(cy_e, py_e)
            iy_max = torch.min(my_e, py_e + pdy_e)
            
            ix_len = torch.clamp(ix_max - ix_min, min=0.0)
            iy_len = torch.clamp(iy_max - iy_min, min=0.0)
            
            support_area = torch.where(is_support, ix_len * iy_len, torch.zeros_like(ix_len))
            total_support_area = support_area.sum(dim=1)
            
            item_area = dx_f * dy_f
            is_on_floor = gravity_z_t <= float(min_z) + 1e-4
            stability_t = is_on_floor | (total_support_area >= 0.5 * item_area - 1e-4)
        else:
            gravity_z_t = torch.zeros(N_t, dtype=torch.float32, device=device)
            stability_t = torch.ones(N_t, dtype=torch.bool, device=device)

        # --- Zone validation (restricted to assigned zone when active) ---
        final_z_t    = torch.full((N_t,), float('inf'), dtype=torch.float32, device=device)
        valid_zone_t = torch.zeros(N_t, dtype=torch.bool, device=device)

        # Accessibility-Aware Zone Assignment:
        # We use the distance from the door to the CLOSEST POINT of the zone
        # (instead of the center) to ensure high-AF items cluster at the front
        # of long zones.
        door_dists = []
        for zne in use_zones:
            zx1, zx2 = float(zne['x1']), float(zne['x2'])
            zy1, zy2 = float(zne['y1']), float(zne['y2'])
            # Closest point in zone to door_x, door_y
            cx_zone = max(zx1, min(zx2, door_x))
            cy_zone = max(zy1, min(zy2, door_y))
            d = math.sqrt((cx_zone - door_x)**2 + (cy_zone - door_y)**2)
            door_dists.append(d)
        
        for zne in valid_zones:
            zx1, zx2 = float(zne['x1']), float(zne['x2'])
            zy1, zy2 = float(zne['y1']), float(zne['y2'])
            zz1 = float(zne.get('z1', 0))
            zz2 = float(zne.get('z2', wh_hgt))

            # Tighter zone boundary tolerance (0.001 m = 1 mm) to prevent items
            # from drifting across zone borders and causing inter-zone overlaps.
            _ZT = 0.001
            in_zone = (
                (all_cx >= zx1 - _ZT) & (max_x_t <= zx2 + _ZT) &
                (all_cy >= zy1 - _ZT) & (max_y_t <= zy2 + _ZT) &
                valid_bounds
            )
            pz_z    = torch.clamp(gravity_z_t, min=max(float(zz1), float(min_z)))
            # Strict Z ceiling: item top must not exceed zone ceiling.
            # into FR territory (zone z=[3,6]) and causing 3-D overlaps.
            fits    = (pz_z + dz_f) <= zz2
            zone_ok = in_zone & fits & stability_t
            update  = zone_ok & (pz_z < final_z_t)
            final_z_t    = torch.where(update, pz_z, final_z_t)
            valid_zone_t = valid_zone_t | zone_ok

        valid_t = valid_zone_t & valid_bounds
        if not valid_t.any():
            continue

        center_x_t = all_cx + dx_f / 2
        center_y_t = all_cy + dy_f / 2
        
        # --- DOOR / WALL / ML HEURISTIC (Vectorized) ---
        d_door_sq_t = (center_x_t - door_x)**2 + (center_y_t - door_y)**2
        
        # Wall distance
        dw_t = torch.full_like(d_door_sq_t, 1e6, device=device)
        for zne in valid_zones:
            zx1_f, zy1_f = float(zne['x1']), float(zne['y1'])
            zx2_f, zy2_f = float(zne['x2']), float(zne['y2'])
            in_this_zone = (all_cx >= zx1_f - 0.01) & (max_x_t <= zx2_f + 0.01) & \
                           (all_cy >= zy1_f - 0.01) & (max_y_t <= zy2_f + 0.01)
            if in_this_zone.any():
                local_dw = torch.min(
                    torch.min(torch.abs(all_cx - zx1_f), torch.abs(all_cx + dx_f - zx2_f)),
                    torch.min(torch.abs(all_cy - zy1_f), torch.abs(all_cy + dy_f - zy2_f))
                )
                dw_t = torch.where(in_this_zone & (local_dw < dw_t), local_dw, dw_t)
        
        d_ml_sq_t = (center_x_t - target_x) ** 2 + (center_y_t - target_y) ** 2
        
        d_door_t = torch.sqrt(d_door_sq_t)
        d_ml_t = torch.sqrt(d_ml_sq_t)
        d_corner_t = torch.sqrt((center_x_t - zx1)**2 + (center_y_t - zy1)**2)
        
        # Heuristic Weights: AF -> Door, Priority -> Wall, ML -> Compactness
        # Harmonized weights: Door max ~500k, Wall max ~500k
        door_attraction = (af ** 1.5) * 15000.0
        wall_attraction = (prio ** 2) * 20000.0 if af >= 5.0 else (prio ** 2) * 50000.0
        ml_weight = 3000.0
        corner_pull = 25000.0
        
        h_score_t = (door_attraction * d_door_t + 
                     wall_attraction * (dw_t ** 2) + 
                     ml_weight * d_ml_t + 
                     corner_pull * d_corner_t)

        # Lexicographic: floor_bias, final_z, h_score
        fz = torch.where(valid_t, final_z_t, BIG)
        _fb_thresh  = max(0.01, float(min_z) + 0.01)
        floor_bias_t = torch.where(fz <= _fb_thresh, torch.zeros_like(fz), torch.ones_like(fz))
        
        # Fast-mode early exit
        if is_fast:
            fc = valid_t & (final_z_t <= _fb_thresh) & (d_ml_sq_t < 0.5)
            if fc.any():
                i = fc.nonzero(as_tuple=True)[0][0].item()
                sc = (final_z_t[i].item(), h_score_t[i].item())
                return (center_x_t[i].item(), center_y_t[i].item(),
                        final_z_t[i].item(), rot_code, dx, dy, dz, sc)

        # ── LEXICOGRAPHICAL SCORING ──
        # For high-AF items (AF >= 5.0), prioritize FLOOR placement, then door
        # proximity, then fine z.  Previously h_score was primary — because all
        # door-proximal columns share near-identical h_score, the first item
        # claimed the single lowest-score column and every subsequent high-AF
        # item stacked on top, overflowing the zone ceiling.  Floor-first makes
        # items fill the door-side row before stacking vertically.
        if af >= 5.0:
            # Score = (Floor-Bias, H-Score, Final-Z)
            min_fb = floor_bias_t.min()
            m = valid_t & (floor_bias_t <= min_fb + 1e-6)
            hs_m = torch.where(m, h_score_t, BIG)
            min_hs = hs_m.min()
            m = m & (hs_m <= min_hs * 1.001 + 0.1)
            fz_m = torch.where(m, fz, BIG)
            bi = fz_m.argmin().item()
            if valid_t[bi]:
                sc   = (floor_bias_t[bi].item(), h_score_t[bi].item(), final_z_t[bi].item())
                cand = (center_x_t[bi].item(), center_y_t[bi].item(),
                        final_z_t[bi].item(), rot_code, dx, dy, dz, sc)
                if best is None or sc < best[7]:
                    best = cand
        else:
            # For low-AF items, prioritize floor placement and ML targets (stability/compactness)
            min_fb = floor_bias_t.min()
            m = valid_t & (floor_bias_t <= min_fb + 1e-6)
            hs_m = torch.where(m, h_score_t, BIG)
            min_hs = hs_m.min()
            m = m & (hs_m <= min_hs * 1.001 + 1.0)
            fz_m = torch.where(m, fz, BIG)
            bi = fz_m.argmin().item()
            if valid_t[bi]:
                sc   = (floor_bias_t[bi].item(), h_score_t[bi].item(), final_z_t[bi].item())
                cand = (center_x_t[bi].item(), center_y_t[bi].item(),
                        final_z_t[bi].item(), rot_code, dx, dy, dz, sc)
                if best is None or sc < best[7]:
                    best = cand

    return best



class ZoneOccupancy:
    """Tracks placed items per zone for NF-top-Z enforcement and touch-point generation."""

    def __init__(self, zones):
        self.zones       = zones
        self.nz          = len(zones)
        self.items       = {zi: [] for zi in range(self.nz)}
        # Initialise NF ceiling at the zone floor — updated as NF items are placed.
        self.max_nf_top  = {zi: float(zones[zi].get('z1', 0)) for zi in range(self.nz)}
        # Fragile ceiling — updated as FR items are placed (for lightweight-on-top enforcement).
        self.max_fr_top  = {zi: float(zones[zi].get('z1', 0)) for zi in range(self.nz)}

    def add(self, zi, x1, y1, z, dx, dy, dz, is_nf=False):
        self.items[zi].append((x1, y1, z, dx, dy, dz))
        if is_nf:
            self.max_nf_top[zi] = max(self.max_nf_top[zi], z + dz)
        else:
            self.max_fr_top[zi] = max(self.max_fr_top[zi], z + dz)

    def touch_points(self, zi, limit=30):
        """Adjacency touch-point (x1, y1) positions from zone zi's recently placed items."""
        pts = []
        recent_items = self.items[zi][-limit:]
        
        x_cands = set()
        y_cands = set()
        for (x1, y1, _z, dx, dy, _dz) in recent_items:
            x_cands.add(x1)
            x_cands.add(x1 + dx)
            y_cands.add(y1)
            y_cands.add(y1 + dy)
            
            # Legacy direct corners
            pts.extend([(x1 + dx, y1), (x1, y1 + dy), (x1 + dx, y1 + dy), (x1, y1)])
            
        # Cross intersections for perfectly tight packing with ZERO gaps
        for cx in x_cands:
            for cy in y_cands:
                pts.append((cx, cy))
                
        return pts


def _perform_search_cpu(cands, rots, l, w, h,
                        placed_items, grid, use_zones, assigned_zone,
                        wh_len, wh_wid, wh_hgt,
                        target_x, target_y, zone_ox, zone_oy,
                        is_fast, min_z=0.0,
                        door_x=0.0, door_y=0.0, af=0.0, prio=1.0,
                        zx1=0.0, zy1=0.0, zx2=0.0, zy2=0.0):
    """CPU fallback item placement search (mirrors _search_candidates_gpu logic).
    Returns (center_x, center_y, z, rot, dx, dy, dz, score) or None."""
    check_zones  = [assigned_zone] if assigned_zone is not None else use_zones
    floor_thresh = max(0.01, float(min_z) + 0.01)
    best         = None

    for rot in rots:
        dx, dy, dz = get_rotated_dims(l, w, h, rot)
        for (cx, cy) in cands:
            max_x, max_y = cx + dx, cy + dy
            if max_x > wh_len + 0.001 or max_y > wh_wid + 0.001:
                continue
            gravity_z = 0.0
            for p_idx in grid.query(cx, cy, max_x, max_y):
                px, py, pz, pdx, pdy, pdz = placed_items[p_idx]
                if (pz < 1000 and max_x > px + 0.001 and cx < px + pdx - 0.001
                        and max_y > py + 0.001 and cy < py + pdy - 0.001):
                    top_z = pz + pdz
                    if top_z > gravity_z:
                        gravity_z = top_z
            valid_z_found = False
            best_z_for_cand = float('inf')
            best_zne_for_cand = None
            
            for zne in check_zones:
                _ZT_CPU = 0.001  # 1 mm zone boundary tolerance (same as GPU path)
                if (cx >= zne['x1'] - _ZT_CPU and max_x <= zne['x2'] + _ZT_CPU and
                        cy >= zne['y1'] - _ZT_CPU and max_y <= zne['y2'] + _ZT_CPU):

                    zne_z2 = zne.get('z2', wh_hgt)
                    actual_gz = 0.0
                    _OVL_CPU = 1e-6  # strict gravity overlap (same as GPU path)
                    for p_idx in grid.query(cx, cy, max_x, max_y):
                        px, py, pz, pdx, pdy, pdz = placed_items[p_idx]
                        if (pz < 1000 and pz < zne_z2 - _OVL_CPU and
                            max_x > px + _OVL_CPU and cx < px + pdx - _OVL_CPU and
                            max_y > py + _OVL_CPU and cy < py + pdy - _OVL_CPU):
                            top_z = pz + pdz
                            if top_z > actual_gz:
                                actual_gz = top_z

                    pz_cand = max(actual_gz, float(zne.get('z1', 0)), float(min_z))
                    # Strict Z ceiling — no +0.001 tolerance (matches GPU path).
                    if pz_cand + dz <= zne_z2 and pz_cand < best_z_for_cand:
                        best_z_for_cand = pz_cand
                        best_zne_for_cand = zne
            
            if best_zne_for_cand is not None:
                final_z = best_z_for_cand
                center_x, center_y = cx + dx / 2, cy + dy / 2
                floor_bias = 0 if final_z <= floor_thresh else 1
                
                # --- DOOR / WALL / ML HEURISTIC ---
                # Weighting: Priority (Wall) > AF (Door) > ML (Compactness)
                d_door_sq = (center_x - door_x)**2 + (center_y - door_y)**2
                # Distance to closest wall of the SELECTED zone
                sz = best_zne_for_cand
                dw = min(abs(cx - sz['x1']), abs(cx + dx - sz['x2']), 
                         abs(cy - sz['y1']), abs(cy + dy - sz['y2']))
                # Distance to ML target
                d_ml_sq = (center_x - target_x)**2 + (center_y - target_y)**2
                
                d_door = math.sqrt(d_door_sq)
                d_ml = math.sqrt(d_ml_sq)
                d_corner = math.sqrt((center_x - sz['x1'])**2 + (center_y - sz['y1'])**2)
                
                # Heuristic Weights: AF -> Door, Priority -> Wall, ML -> Compactness
                # Harmonized weights: Door max ~500k, Wall max ~500k
                door_attraction = (af ** 1.5) * 15000.0
                wall_attraction = (prio ** 2) * 20000.0 if af >= 5.0 else (prio ** 2) * 50000.0
                ml_weight = 3000.0
                corner_pull = 25000.0
                
                h_score = (door_attraction * d_door + 
                           wall_attraction * (dw ** 2) + 
                           ml_weight * d_ml + 
                           corner_pull * d_corner)
                
                # High-AF: floor-first, then door-proximity, then fine z —
                # keeps items from piling at the lowest-h_score column.
                if af >= 5.0:
                    score = (floor_bias, h_score, final_z)
                else:
                    # Low-AF: prioritize floor placement and stability
                    score = (floor_bias, final_z, h_score)

                if best is None or score < best[7]:
                    best = (center_x, center_y, final_z, rot, dx, dy, dz, score)
                if is_fast and final_z <= floor_thresh and d_ml_sq < 0.1:
                    return best
    return best


def repair_solution_compact(solution, items_props, warehouse_dims, allocation_zones=None, layer_heights=None, callback=None, callback_interval=50, fast_mode=False, max_candidates=None, item_categories=None, item_names=None, warehouse_id=None, warehouse_name=None):
    print(f'>>> repair_solution_compact CALLED.')
    """Repair solution by placing items in valid positions with gravity."""
    # Defaults
    if layer_heights is None or len(layer_heights) == 0:
        layer_heights = [0.0]
    
    wh_len = warehouse_dims[0] if warehouse_dims else 100
    wh_wid = warehouse_dims[1] if warehouse_dims else 100
    wh_hgt = warehouse_dims[2] if warehouse_dims else 10
    # Door coordinates for accessibility-aware heuristic
    door_x = warehouse_dims[3] if warehouse_dims and len(warehouse_dims) > 3 else 0.0
    door_y = warehouse_dims[4] if warehouse_dims and len(warehouse_dims) > 4 else 0.0
    
    num_items = len(solution)
    if num_items == 0: return solution

    # Sort indices for heuristic placement:
    # 1. Fragility (Ascending) - 0 (Robust) first, 1 (Fragile) last
    # 2. Access Frequency (Descending) - High-traffic items first (closest to door)
    # 3. Priority (Descending) - Important items secure stable bases
    # 4. Weight (Descending) - Heavy items for base stability
    # 5. Volume (Descending) - Large items to fill gaps
    fragility = items_props[:, 8]
    access_freqs = items_props[:, 5]
    priorities = items_props[:, 9] if items_props.shape[1] > 9 else np.ones(num_items)
    weights = items_props[:, 6]
    volumes = items_props[:, 0] * items_props[:, 1] * items_props[:, 2]
    
    indices = np.arange(num_items)
    use_zones = allocation_zones if allocation_zones else [{'x1':0, 'y1':0, 'x2':wh_len, 'y2':wh_wid, 'z1':0, 'z2':wh_hgt}]
    
    # Sort zones by Z-base (Bottom first), then Y1, then X1 for back-to-front, bottom-to-top filling
    use_zones = sorted(use_zones, key=lambda zn: (zn.get('z1', 0), zn.get('y1', 0), zn.get('x1', 0)))
    
    # Map item categories to specific zones by name (case-insensitive)
    cat_to_zones = {}
    if item_categories:
        unique_cats = set(c for c in item_categories if c)
        for zi, zne in enumerate(use_zones):
            zname = zne.get('name', '').lower()
            for cat in unique_cats:
                if cat.lower() in zname:
                    if cat not in cat_to_zones: cat_to_zones[cat] = []
                    cat_to_zones[cat].append(zi)
    
    num_zones = len(use_zones)

    if num_zones >= 2:
        # Separate fragile vs non-fragile, each sorted heavy-first for stability
        fragile_indices = [i for i in indices if fragility[i] == 1]
        non_fragile_indices = [i for i in indices if fragility[i] != 1]
        
        fragile_list = sorted(fragile_indices, key=lambda i: (-weights[i], -access_freqs[i], -priorities[i], -volumes[i], i))
        non_fragile_list = sorted(non_fragile_indices, key=lambda i: (-access_freqs[i], -priorities[i], -weights[i], -volumes[i], i))

        # --- Look-Ahead Capacity-Aware Multi-Zone Assignment ---
        #      receive fragile overflow.
        #   2. Pre-reserve the smallest NF items as a base layer in those upper zones
        #      BEFORE filling the bottom zone with the remaining NF items.
        #   3. Fill the bottom zone maximally with the remaining (non-reserved) NF items.
        #   4. Assign fragile items — overflow lands on top of the pre-existing NF base.
        #   5. Build stacking order: [all NF zones in order] then [all F zones in order].
        #      This guarantees every zone has NF placed before F, regardless of level.
        #
        unique_z_levels = sorted(list(set(zne.get('z1', 0) for zne in use_zones)))
        zones_by_lvl = {
            lvl: [zi for zi, zne in enumerate(use_zones) if zne.get('z1', 0) == lvl]
            for lvl in unique_z_levels
        }

        # Buffer for zone-assignment log lines (written after _log_f is opened)
        _zone_assign_log = []

        # Pre-compute per-zone capacity and floor area once.
        zone_caps = {}
        zone_floor_areas = {}
        for zi, zne in enumerate(use_zones):
            _dx = zne['x2'] - zne['x1']
            _dy = zne['y2'] - zne['y1']
            _dz = zne['z2'] - zne['z1']
            zone_caps[zi]        = _dx * _dy * _dz
            zone_floor_areas[zi] = _dx * _dy

        zone_used_vols = {zi: 0.0 for zi in range(num_zones)}
        item_zone_idx  = {}

        # ── Helper: advance level index while current level >= 85% full ──
        def _overflow_level(used_vols, l_idx):
            while l_idx < len(unique_z_levels) - 1:
                lvl     = unique_z_levels[l_idx]
                lvl_cap  = sum(zone_caps[zi] for zi in zones_by_lvl[lvl])
                lvl_used = sum(used_vols[zi]  for zi in zones_by_lvl[lvl])
                if lvl_cap > 0 and lvl_used >= lvl_cap * 0.85:
                    l_idx += 1
                else:
                    break
            return l_idx

        # ── Helper: pick best zone within a level ──
        def _best_zone_in_level(item_i, lvl_idx, used_vols):
            lvl          = unique_z_levels[lvl_idx]
            target_zones = zones_by_lvl[lvl]

            # --- CATEGORY MATCHING ---
            # If item has a category-specific zone at this level, use only those.
            if item_categories:
                item_cat = item_categories[item_i]
                matching_zone_indices = cat_to_zones.get(item_cat, [])
                if matching_zone_indices:
                    cat_targets = [zi for zi in target_zones if zi in matching_zone_indices]
                    if cat_targets:
                        target_zones = cat_targets
            # -------------------------

            l, w, h  = items_props[item_i, 0:3]
            af_i     = access_freqs[item_i]
            prio_i   = priorities[item_i]
            eligible = []
            for zi in target_zones:
                zne   = use_zones[zi]
                max_d = max(l, w, h)
                z_max = max(zne['x2'] - zne['x1'],
                            zne['y2'] - zne['y1'],
                            zne['z2'] - zne['z1'])
                if max_d <= z_max + 0.1:
                    cap     = zone_caps[zi]
                    rem_pct = (cap - used_vols[zi]) / cap if cap > 0 else 0
                    # Distance from zone centroid to the warehouse door —
                    # used to route high-AF items to door-proximal zones.
                    zne_cx   = (zne['x1'] + zne['x2']) / 2.0
                    zne_cy   = (zne['y1'] + zne['y2']) / 2.0
                    door_dist = math.sqrt((zne_cx - door_x) ** 2 +
                                          (zne_cy - door_y) ** 2)
                    eligible.append((zi, rem_pct, door_dist))
            if eligible:
                if af_i >= 5.0:
                    # High access-frequency: prefer zones closest to the door,
                    # load-balancing across all zones within a distance tolerance
                    # so parallel corridors are filled in tandem rather than one
                    # corridor being exhausted before the next is touched.
                    has_space = [e for e in eligible if e[1] > 0.05]
                    pool = has_space if has_space else eligible
                    min_door_dist = min(e[2] for e in pool)
                    # Accept any zone within 1.5 m (or 15%) of the nearest zone.
                    # This prevents a trivial centroid-distance tie (e.g. 10.04 vs
                    # 10.06 m) from routing all items to a single corridor.
                    tolerance = max(1.5, min_door_dist * 0.15)
                    near_pool = [e for e in pool if e[2] <= min_door_dist + tolerance]
                    # Among equidistant zones, pick the one with the most remaining
                    # capacity so items spread evenly across parallel corridors.
                    chosen_zi = max(near_pool, key=lambda e: e[1])[0]
                    # Buffer zone assignment log (flushed after _log_f is opened)
                    chosen_dist = next(e[2] for e in eligible if e[0] == chosen_zi)
                    chosen_rem  = next(e[1] for e in eligible if e[0] == chosen_zi)
                    _zone_assign_log.append(
                        f'  [ZONE-ASSIGN] item={item_i} af={af_i:.0f} '
                        f'-> zone={chosen_zi} door_dist={chosen_dist:.2f}m '
                        f'rem={chosen_rem:.2%} '
                        f'eligible={[(e[0], round(e[2],2), f"{e[1]:.0%}") for e in eligible]}'
                        f'{"  [NO-SPACE-FALLBACK]" if not has_space else ""}\n'
                    )
                    return chosen_zi
                else:
                    # Standard items: balance load across zones.
                    random.shuffle(eligible)
                    return max(eligible, key=lambda e: e[1])[0]
            # Fallback: round-robin within level
            return target_zones[item_i % len(target_zones)]

        # ── PHASE 1: Predict Warehouse Occupancy and Target Levels ──
        total_nf_vol = sum(volumes[i] for i in non_fragile_indices)
        total_fr_vol = sum(volumes[i] for i in fragile_indices)
        total_vol    = total_nf_vol + total_fr_vol
        
        # FR items start at level 0 (bottom-first filling). NF items are always
        # placed first in each zone, ensuring FR naturally stacks on top of NF.
        _fr_start_idx = 0
        sim_used = {zi: 0.0 for zi in range(num_zones)}
        max_occupied_l_idx = 0
        # NF items fill from level 0 up
        sim_nf_idx = 0
        for item_i in non_fragile_list:
            sim_nf_idx = _overflow_level(sim_used, sim_nf_idx)
            best_zi = _best_zone_in_level(item_i, sim_nf_idx, sim_used)
            sim_used[best_zi] += volumes[item_i]
            max_occupied_l_idx = max(max_occupied_l_idx, sim_nf_idx)
        # FR items fill from first upper level up
        sim_fr_idx = _fr_start_idx
        for item_i in fragile_list:
            sim_fr_idx = _overflow_level(sim_used, sim_fr_idx)
            best_zi = _best_zone_in_level(item_i, sim_fr_idx, sim_used)
            sim_used[best_zi] += volumes[item_i]
            max_occupied_l_idx = max(max_occupied_l_idx, sim_fr_idx)

        # ── PHASE 2 & 3: Greedy Bottom-Up Allocation ──
        # We fill levels sequentially (0, 1, 2...) to 100% capacity.
        # This ensures the bottom zones are 'filled up first' as requested.
        item_zone_idx = {}
        zone_used_vols = {zi: 0.0 for zi in range(num_zones)}
        
        print(f"DEBUG shelf: Greedy Bottom-Up active. Total Items={num_items}")
        
        # 1. Non-Fragile Items
        l_idx = 0
        for item_i in non_fragile_list:
            l_idx = _overflow_level(zone_used_vols, l_idx)
            best_zi = _best_zone_in_level(item_i, l_idx, zone_used_vols)
            item_zone_idx[item_i] = best_zi
            zone_used_vols[best_zi] += volumes[item_i]
            
        # 2. Fragile Items — start at the FIRST UPPER level so every FR item is
        #    physically above all NF items (fragility constraint: fragile on top).
        #    Within the fragile group, heavy FR items are sorted first so they
        #    occupy the bottom of the upper zone; lightweight FR items are sorted
        #    last and will be stacked on top by gravity (lightweight-on-top rule).
        l_idx = _fr_start_idx
        for item_i in fragile_list:
            l_idx = _overflow_level(zone_used_vols, l_idx)
            best_zi = _best_zone_in_level(item_i, l_idx, zone_used_vols)
            item_zone_idx[item_i] = best_zi
            zone_used_vols[best_zi] += volumes[item_i]

        # ── DIAGNOSTIC: zone-assignment distribution (buffered for _log_f) ──
        _zone_diag_lines = []
        _zone_diag_lines.append(
            f'--- ZONE ASSIGN DIAG: unique_z_levels={unique_z_levels}  '
            f'_fr_start_idx={_fr_start_idx}  num_zones={num_zones} ---\n'
        )
        for _zi in range(num_zones):
            _zne = use_zones[_zi]
            _nf_c = sum(1 for i, zi in item_zone_idx.items() if zi == _zi and fragility[i] != 1)
            _fr_c = sum(1 for i, zi in item_zone_idx.items() if zi == _zi and fragility[i] == 1)
            _zone_diag_lines.append(
                f'  zone={_zi} z1={_zne.get("z1",0):.2f} z2={_zne.get("z2",wh_hgt):.2f}  '
                f'NF_assigned={_nf_c}  FR_assigned={_fr_c}  '
                f'used_vol={zone_used_vols[_zi]:.2f}/{zone_caps[_zi]:.2f}\n'
            )
        # Prepend diag to the zone-assign log so it appears first in the file.
        _zone_assign_log = _zone_diag_lines + _zone_assign_log

        # ── PHASE 4: Build Stacking Order (NF-First, then composite AF×Prio) ──
        # Composite AF×priority ensures the highest-value items (both frequent AND
        # important) get first pick of door-proximal positions in both layers.
        sorted_indices = sorted(indices, key=lambda i: (
            fragility[i],                                                        # NF (0) always before FR (1)
            -weights[i] if fragility[i] == 1 else -((access_freqs[i] // 2) * 2), # FR: Sort by Weight first. NF: Sort by AF first
            -((access_freqs[i] // 2) * 2) if fragility[i] == 1 else -weights[i], # Secondary sort
            -access_freqs[i],                                                    # Specific AF tie-break
            -volumes[i],                                                         # Volume tie-break
            i
        ))

        print(
            f"DEBUG look-ahead: Proportional stacking active across {max_occupied_l_idx + 1} levels. "
            f"NF Ratio: {total_nf_vol/max(total_vol, 1e-6):.2%}."
        )

    else:
        # Single zone: build NF/F lists and treat entire warehouse as zone 0
        non_fragile_indices = [i for i in indices if fragility[i] != 1]
        fragile_indices     = [i for i in indices if fragility[i] == 1]
        non_fragile_list    = sorted(non_fragile_indices, key=lambda i: (-access_freqs[i], -priorities[i], -weights[i], -volumes[i], i))
        fragile_list        = sorted(fragile_indices,     key=lambda i: (-access_freqs[i], -priorities[i], -weights[i], -volumes[i], i))
        item_zone_idx       = {i: 0 for i in indices}

    # ─────────────────────────────────────────────────────────────────────────
    # Physical Placement Infrastructure
    # ─────────────────────────────────────────────────────────────────────────
    zone_occ     = ZoneOccupancy(use_zones)
    placed_items = []                           # global list: (x1, y1, z, dx, dy, dz)
    global_grid  = SimpleGrid(wh_len, wh_wid, cell_size=0.5) # Balanced density: 0.5m cells for items 0.3-1.5m

    if _TORCH_AVAILABLE:
        _dev          = _TORCH_DEVICE
        _placed_buf   = _torch.zeros((num_items, 6), dtype=_torch.float32, device=_dev)
        _placed_count = 0
    else:
        _dev = _placed_buf = None
        _placed_count = 0

    # ── Placement debug log (written to placement_debug.log) ──────────────────
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'placement_debug.log')
    _log_f    = open(_log_path, 'w', encoding='utf-8')
    _ts       = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    _log_f.write(f'\n{"="*80}\n')
    _log_f.write(f'repair_solution_compact  [{_ts}]\n')
    
    _wh_id_str = f" ID={warehouse_id}" if warehouse_id is not None else ""
    _wh_name_str = f" '{warehouse_name}'" if warehouse_name else ""
    _log_f.write(f'WAREHOUSE CONFIGURATION{_wh_id_str}{_wh_name_str}\n')
    _log_f.write(f'  DIMENSIONS: {wh_len:.2f}m (L) x {wh_wid:.2f}m (W) x {wh_hgt:.2f}m (H)\n')
    _log_f.write(f'  VOLUME:     {wh_len*wh_wid*wh_hgt:.2f} m^3\n')
    
    # Determine Door Wall Location for clear diagnostics
    _door_loc = "Custom/Internal"
    if door_x <= 0.01: _door_loc = "Left Wall (X=0)"
    elif abs(door_x - wh_len) <= 0.01: _door_loc = "Right Wall (X=L)"
    elif door_y <= 0.01: _door_loc = "Front Wall (Y=0)"
    elif abs(door_y - wh_wid) <= 0.01: _door_loc = "Back Wall (Y=W)"
    
    _log_f.write(f'  DOOR LOC:   ({door_x:.2f}, {door_y:.2f}) -> {_door_loc}\n')
    _log_f.write(f'  HEURISTIC:  Accessibility-Aware (AF pulls to Door, Prio pulls to Wall)\n')
    
    # --- Logging: Pre-Placement Audit ---
    _log_f.write('--- PRE-PLACEMENT AUDIT: Top 10 Priority Items ---\n')
    all_sorted = sorted(indices, key=lambda i: (fragility[i], -access_freqs[i], -priorities[i], -weights[i], -volumes[i], i))
    for i in all_sorted[:10]:
        _log_f.write(f'  Rank={all_sorted.index(i)+1:>2d} | item={i:>4d} af={access_freqs[i]:.0f} prio={priorities[i]:.0f} frag={fragility[i]}\n')
    _log_f.write('--------------------------------------------------\n')
    _log_f.flush()

    _log_f.write(f'  items={num_items}  nf={len(non_fragile_list)}  fr={len(fragile_list)}  total_item_vol={float(np.sum(volumes)):.2f}  theoretical_util={(float(np.sum(volumes))/(wh_len*wh_wid*wh_hgt if wh_len*wh_wid*wh_hgt > 0 else 1)*100):.1f}%\n'); _log_f.flush()

    # ── Warehouse dimensions ──────────────────────────────────────────────────
    _wh_vol = wh_len * wh_wid * wh_hgt
    _item_vol_total = float(np.sum(volumes))

    # ── Zone dimensions ───────────────────────────────────────────────────────
    _log_f.write(f'ZONES  count={num_zones}\n')
    for _zi, _zne in enumerate(use_zones):
        _zdx  = _zne['x2'] - _zne['x1']
        _zdy  = _zne['y2'] - _zne['y1']
        _zdz  = _zne.get('z2', wh_hgt) - _zne.get('z1', 0)
        _zvol = _zdx * _zdy * _zdz
        _zarea = _zdx * _zdy
        _znf_cnt = sum(1 for i, zi in (item_zone_idx or {}).items() if zi == _zi and fragility[i] != 1)
        _zfr_cnt = sum(1 for i, zi in (item_zone_idx or {}).items() if zi == _zi and fragility[i] == 1)
        _znf_vol = sum(float(volumes[i]) for i, zi in (item_zone_idx or {}).items() if zi == _zi and fragility[i] != 1)
        _zfr_vol = sum(float(volumes[i]) for i, zi in (item_zone_idx or {}).items() if zi == _zi and fragility[i] == 1)
        _zname   = _zne.get('name', _zne.get('label', f'zone_{_zi}'))
        _ztype   = _zne.get('zone_type', 'allocation')
        _log_f.write(
            f'  Zone {_zi} "{_zname}" [{_ztype}]\n'
            f'    x=[{_zne["x1"]:.2f}, {_zne["x2"]:.2f}]  '
            f'y=[{_zne["y1"]:.2f}, {_zne["y2"]:.2f}]  '
            f'z=[{_zne.get("z1",0):.2f}, {_zne.get("z2",wh_hgt):.2f}]\n'
            f'    size=({_zdx:.2f} x {_zdy:.2f} x {_zdz:.2f})  '
            f'floor_area={_zarea:.2f}  volume={_zvol:.2f}\n'
            f'    assigned: {_znf_cnt} NF ({_znf_vol:.2f} vol) + '
            f'{_zfr_cnt} FR ({_zfr_vol:.2f} vol)  '
            f'load={(_znf_vol+_zfr_vol)/_zvol*100:.1f}%\n'
        )
    # Flush buffered zone-assign log lines (from PHASE 1/2/3, before log was open)
    _zon_log = locals().get('_zone_assign_log', [])
    if _zon_log:
        _log_f.write(f'--- ZONE ASSIGNMENT LOG ({len(_zon_log)} high-AF items) ---\n')
        _log_f.writelines(_zon_log)
    _log_f.flush()


    def _log_item(idx, pass_name, zi, is_nf, min_z, b_x, b_y, b_z, b_dx, b_dy, b_dz, used_fallback):
        frag_label = 'NF' if is_nf else 'FR'
        fb_flag    = ' [FALLBACK]' if used_fallback else ''

        # Enhanced item details
        name = f"({item_names[idx]})" if item_names and idx < len(item_names) else ""
        af = items_props[idx, 5]
        prio = int(items_props[idx, 9]) if items_props.shape[1] > 9 else 1
        
        # Heuristic Components for Debugging
        d_door_sq = (b_x - door_x) ** 2 + (b_y - door_y) ** 2
        d_door = math.sqrt(d_door_sq)
        
        # Re-calculate wall distance for logging
        dw = 1e6
        if zi is not None:
            sz = use_zones[zi]
            dw = min(abs(b_x - b_dx/2 - sz['x1']), abs(b_x + b_dx/2 - sz['x2']), 
                     abs(b_y - b_dy/2 - sz['y1']), abs(b_y + b_dy/2 - sz['y2']))
        
        # Recalculate forces using HARMONIZED weights for diagnostic accuracy
        door_attraction_val = (af ** 1.5) * 15000.0
        wall_attraction_val = (prio ** 2) * 20000.0 if af >= 5.0 else (prio ** 2) * 50000.0
        ml_weight_val = 3000.0

        door_force = door_attraction_val * d_door
        wall_force = wall_attraction_val * (dw ** 2)
        # Initial target for comparison
        tx, ty = solution[idx, 0], solution[idx, 1]
        
        ml_dist    = math.sqrt((b_x - tx)**2 + (b_y - ty)**2)
        ml_force   = ml_weight_val * ml_dist
        
        _log_f.write(
            f'  [NEW_LOG_V3] {pass_name} | item={idx:>4d} {name:<20} {frag_label} '
            f'wt={weights[idx]:>4.1f} prio={prio} af={af:.0f} zone={zi} '
            f'xyz=({b_x:>5.2f},{b_y:>5.2f},{b_z:>5.2f}) '
            f'ml=({tx:>5.2f},{ty:>5.2f}) '
            f'd_door={d_door:>5.2f}m dw={dw:>5.2f}m '
            f'forces=(D:{door_force/1000.0:>6.1f}k, W:{wall_force/1000.0:>6.1f}k, ML:{ml_force/1000.0:>5.1f}k) '
            f'{fb_flag}\n'
        )


    # ── Single-item placement core ────────────────────────────────────────────
    def _place_item(idx, assigned_zi, min_z, is_nf):
        nonlocal _placed_count

        l, w, h    = items_props[idx, 0:3]
        can_rotate = int(items_props[idx, 3])
        rots       = list(range(6)) if can_rotate else [int(solution[idx, 3])]
        az         = use_zones[assigned_zi] if assigned_zi is not None else None
        tx         = float(solution[idx, 0])
        ty         = float(solution[idx, 1])
        pc         = _placed_count  # snapshot for GPU buffer slice

        # ── Candidate generation: zone-origin + ML warm-start + touch-points ──
        cands = set()
        if az is not None:
            zx1, zy1 = float(az['x1']), float(az['y1'])
            zx2, zy2 = float(az['x2']), float(az['y2'])
            # Ensure we always check the 4 corners of the zone for wall-hugging
            cands.add((zx1, zy1))
            cands.add((max(zx1, zx2 - l), zy1))
            cands.add((zx1, max(zy1, zy2 - w)))
            cands.add((max(zx1, zx2 - l), max(zy1, zy2 - w)))
            
            # Add projections onto the 4 walls to allow hugging anywhere along the perimeter
            # Left/Right walls
            ty_clamped = max(zy1, min(zy2 - w, ty - w/2))
            cands.add((zx1, ty_clamped))
            cands.add((max(zx1, zx2 - l), ty_clamped))
            # Top/Bottom walls
            tx_clamped = max(zx1, min(zx2 - l, tx - l/2))
            cands.add((tx_clamped, zy1))
            cands.add((tx_clamped, max(zy1, zy2 - w)))
            
            cands.add((max(zx1, min(zx2 - l, tx - l/2)),
                       max(zy1, min(zy2 - w, ty - w/2))))
            for pt in zone_occ.touch_points(assigned_zi):
                cands.add(pt)

            # ── AF-aware: seed positions along the door-closest zone edge ────
            # Determine which wall of this zone is nearest the door and
            # densely sample it so high-AF items always compete for those spots.
            _af_now   = float(items_props[idx, 5])
            _prio_now = float(items_props[idx, 9]) if items_props.shape[1] > 9 else 1.0
            if _af_now >= 5.0:
                _edge_dists = {
                    'x1': abs(zx1 - door_x), 'x2': abs(zx2 - door_x),
                    'y1': abs(zy1 - door_y), 'y2': abs(zy2 - door_y),
                }
                _near_edge = min(_edge_dists, key=_edge_dists.get)
                _cands_before = len(cands)
                _step = 0.5
                
                # ── UNIVERSAL AUTO-ADAPT DOOR LOGIC ──
                # 1. Determine which of the 4 warehouse walls the door is closest to
                dist_to_walls = {
                    'left':  abs(door_x - 0),
                    'right': abs(door_x - wh_len),
                    'front': abs(door_y - 0),
                    'back':  abs(door_y - wh_wid)
                }
                best_wh_wall = min(dist_to_walls, key=dist_to_walls.get)
                
                # 2. Sample the zone edge that faces that specific warehouse wall
                if best_wh_wall == 'left':
                    edge_x = zx1
                    for ey in np.linspace(zy1, zy2 - w, 30): cands.add((edge_x, round(ey, 4)))
                elif best_wh_wall == 'right':
                    edge_x = max(zx1, zx2 - l)
                    for ey in np.linspace(zy1, zy2 - w, 30): cands.add((edge_x, round(ey, 4)))
                elif best_wh_wall == 'front':
                    edge_y = zy1
                    for ex in np.linspace(zx1, zx2 - l, 30): cands.add((round(ex, 4), edge_y))
                else: # back
                    edge_y = max(zy1, zy2 - w)
                    for ex in np.linspace(zx1, zx2 - l, 30): cands.add((round(ex, 4), edge_y))

                _log_f.write(
                    f'  [AUTO-ADAPT-DOOR] item={idx} af={_af_now:.0f} zone={assigned_zi} '
                    f'door_at=({door_x:.2f},{door_y:.2f}) closest_wall={best_wh_wall} '
                    f'added={len(cands)-_cands_before} dynamic candidates\n'
                )

            # ── Priority-aware: sample all four zone walls for high-prio items ─
            # Drives priority >= 2 items flush against any zone wall.
            if _prio_now >= 2.0:
                _step = 0.5
                _cx = zx1
                while _cx <= zx2 - l + 0.01:
                    cands.add((round(_cx, 4), zy1))               # y1 wall
                    cands.add((round(_cx, 4), max(zy1, zy2 - w))) # y2 wall
                    _cx += _step
                _cy = zy1
                while _cy <= zy2 - w + 0.01:
                    cands.add((zx1, round(_cy, 4)))               # x1 wall
                    cands.add((max(zx1, zx2 - l), round(_cy, 4))) # x2 wall
                    _cy += _step

            zone_ox, zone_oy = zx1, zy1
        else:
            for zne in use_zones:
                cands.add((float(zne['x1']), float(zne['y1'])))
            cands.add((tx - l/2, ty - w/2))
            zx1, zy1, zx2, zy2 = 0.0, 0.0, wh_len, wh_wid
            zone_ox, zone_oy   = 0.0, 0.0

        # Only add global adjacency when there is NO assigned zone (no-zone / fallback path).
        # When az is set, zone_occ.touch_points already provides all relevant adjacency candidates
        # from items inside the same zone. Cross-zone touch-points would pollute the candidate
        # set with positions outside this zone's bounds.
        # Generate tight-fit candidates from ALL items in the same zone
        # to eliminate gaps (air bubbles) at the zone level.
        if assigned_zi is not None:
            zone_pts = zone_occ.touch_points(assigned_zi)
            for (tx, ty) in zone_pts:
                cands.add((tx, ty))
                cands.add((tx - l, ty)) # flush-right
                cands.add((tx, ty - w)) # back-align
        
        # Cross-zone/Warehouse candidates (limited for performance)
        for (px, py, _pz, pdx, pdy, _pdz) in placed_items[-50:]:
            cands.add((px + pdx, py))
            cands.add((px, py + pdy))

        valid_cands = [(cx, cy) for (cx, cy) in cands
                       if 0 <= cx < wh_len and 0 <= cy < wh_wid]

        sort_tx = max(zx1, min(zx2 - l, tx - l/2))
        sort_ty = max(zy1, min(zy2 - w, ty - w/2))

        # --- ATTRACTION HEURISTIC ---
        # 1. Access Frequency (AF) pulls items towards the door (x=door_x, y=door_y)
        # 2. Priority pulls items towards the walls of the zone (zx1, zy1, zx2, zy2)
        # 3. Compactness pulls items towards the ML-predicted target (sort_tx, sort_ty)
        af   = float(items_props[idx, 5])
        prio = float(items_props[idx, 9]) if items_props.shape[1] > 9 else 1.0

        def _get_score(p):
            cx, cy = p
            # Center of the item for door distance
            c_x, c_y = cx + l/2, cy + w/2
            # Door distance (weighted by AF)
            d_door_sq = (c_x - door_x)**2 + (c_y - door_y)**2
            # Wall distance (weighted by Priority)
            # Item bounds: [cx, cx+l], [cy, cy+w]. Zone bounds: [zx1, zx2], [zy1, zy2]
            dw = min(abs(cx - zx1), abs(cx + l - zx2), abs(cy - zy1), abs(cy + w - zy2))
            
            # Use IDENTICAL weights to _perform_search_cpu to prevent truncation of good candidates
            # Harmonized weights: Door max ~500k, Wall max ~500k
            # We use LINEAR d_door to prevent squaring from dominating the wall pull.
            door_attraction = (af ** 1.5) * 15000.0
            wall_attraction = (prio ** 2) * 20000.0 if af >= 5.0 else (prio ** 2) * 50000.0
            ml_weight = 3000.0
            # Strong corner pull: towards the zone's front-left origin to eliminate bubbles
            corner_pull = 25000.0
            
            d_door = math.sqrt(d_door_sq)
            d_corner = math.sqrt((cx - zx1)**2 + (cy - zy1)**2)
            
            return (door_attraction * d_door + 
                    wall_attraction * (dw ** 2) + 
                    ml_weight * math.sqrt((cx - sort_tx)**2 + (cy - sort_ty)**2) +
                    corner_pull * d_corner)

        sorted_cands = sorted(valid_cands, key=_get_score)

        if af >= 5.0:
            cand_limit = 20000
        elif max_candidates is not None:
            cand_limit = max_candidates
        elif az is not None or (not fast_mode and volumes[idx] > 0.5):
            cand_limit = 10000
        else:
            # Aggressive throttling for EO-GA fast_mode
            cand_limit = 800 if fast_mode else 5000
        search_cands = sorted_cands[:cand_limit]

        def _search(cand_list, is_fast, search_az=az):
            if _TORCH_AVAILABLE:
                return _search_candidates_gpu(
                    cand_list, rots, l, w, h,
                    _placed_buf, pc, use_zones, search_az,
                    wh_len, wh_wid, wh_hgt,
                    sort_tx + l/2, sort_ty + w/2,
                    is_fast, _dev, min_z=min_z,
                    door_x=door_x, door_y=door_y, af=af, prio=prio,
                    zx1=zx1, zy1=zy1, zx2=zx2, zy2=zy2)
            
            return _perform_search_cpu(
                cand_list, rots, l, w, h,
                placed_items, global_grid, use_zones, search_az,
                wh_len, wh_wid, wh_hgt,
                sort_tx + l/2, sort_ty + w/2, zone_ox, zone_oy,
                is_fast, min_z=min_z,
                door_x=door_x, door_y=door_y, af=af, prio=prio,
                zx1=zx1, zy1=zy1, zx2=zx2, zy2=zy2)

        best_pos = _search(search_cands, fast_mode)

        # Log winning position for high-AF items before fallback
        if af >= 5.0 and best_pos is not None:
            _wx, _wy = best_pos[0], best_pos[1]
            _wd = math.sqrt((_wx - door_x)**2 + (_wy - door_y)**2)
            _log_f.write(
                f'  [PLACE-WIN] item={idx} af={af:.0f} zone={assigned_zi} '
                f'winner=({_wx:.3f},{_wy:.3f}) d_door={_wd:.2f}m '
                f'score={best_pos[7]}\n'
            )

        # ── Lattice fallback: 0.25 → 0.10 → 0.05 m ──────────────────────────
        if best_pos is None:
            rx1 = float(az['x1']) if az else 0.0
            ry1 = float(az['y1']) if az else 0.0
            rx2 = float(az['x2']) if az else wh_len
            ry2 = float(az['y2']) if az else wh_wid
            # Skip the densest lattice (0.05m) in fast_mode to prevent hanging
            lattice_steps = [0.25, 0.10] if fast_mode else [0.25, 0.10, 0.05]
            for step in lattice_steps:
                if best_pos is not None:
                    break
                retry_cands = set()
                cx = rx1
                while cx <= rx2 + step * 0.5:
                    cy = ry1
                    while cy <= ry2 + step * 0.5:
                        retry_cands.add((cx, cy))
                        cy = round(cy + step, 6)
                    cx = round(cx + step, 6)
                # Adjacency touch-points from zone's own items only (strict independence)
                zone_items = zone_occ.items[assigned_zi] if assigned_zi is not None else []
                for (px, py, _pz, pdx, pdy, _pdz) in zone_items:
                    retry_cands.update([
                        (px + pdx, py), (px, py + pdy),
                        (px + pdx, py + pdy), (px, py),
                        (px + pdx - l, py), (px, py + pdy - w),
                    ])
                valid_retry = [
                    (cx, cy) for (cx, cy) in retry_cands
                    if cx >= rx1 - 0.001 and cy >= ry1 - 0.001
                    and cx + l <= rx2 + 0.001 and cy + w <= ry2 + 0.001
                    and cx >= 0 and cy >= 0
                ]
                if valid_retry:
                    # In lattice fallback, still prioritize proximity to the door
                    retry_sorted = sorted(valid_retry,
                                          key=lambda p: (p[0]-door_x)**2 + (p[1]-door_y)**2)
                    best_pos = _search(retry_sorted, False, search_az=az)

        # ── Cross-zone overflow (fragility-safe) ─────────────────────────────
        # If the assigned zone is geometrically full, try other zones at the
        # SAME z1 level so fragility is preserved (NF stays in NF-tier zones,
        # FR stays in FR-tier zones).
        if best_pos is None and num_zones > 1 and az is not None:
            orig_z1 = float(az.get('z1', 0))
            for alt_zi, alt_zne in enumerate(use_zones):
                if alt_zi == assigned_zi:
                    continue
                if abs(float(alt_zne.get('z1', 0)) - orig_z1) > 0.01:
                    continue
                alt_zx1, alt_zy1 = float(alt_zne['x1']), float(alt_zne['y1'])
                alt_zx2, alt_zy2 = float(alt_zne['x2']), float(alt_zne['y2'])
                alt_cands = set()
                step = 0.25
                _cx = alt_zx1
                while _cx <= alt_zx2 - l + 0.01:
                    _cy = alt_zy1
                    while _cy <= alt_zy2 - w + 0.01:
                        alt_cands.add((round(_cx, 4), round(_cy, 4)))
                        _cy = round(_cy + step, 4)
                    _cx = round(_cx + step, 4)
                for (px, py, _pz, pdx, pdy, _pdz) in zone_occ.items[alt_zi]:
                    alt_cands.update([(px + pdx, py), (px, py + pdy)])
                alt_valid = [
                    (cx, cy) for (cx, cy) in alt_cands
                    if cx >= 0 and cy >= 0
                    and cx + l <= wh_len + 0.001 and cy + w <= wh_wid + 0.001
                ]
                if not alt_valid:
                    continue
                alt_sorted = sorted(alt_valid,
                                    key=lambda p: (p[0] - door_x) ** 2 + (p[1] - door_y) ** 2)
                best_pos = _search(alt_sorted, False, search_az=alt_zne)
                if best_pos is not None:
                    _log_f.write(
                        f'  [CROSS-ZONE-OVERFLOW] item={idx} '
                        f'from_zone={assigned_zi} -> alt_zone={alt_zi} '
                        f'(z1={orig_z1:.2f}, fragility-safe)\n'
                    )
                    assigned_zi = alt_zi
                    az = alt_zne
                    break

        # ── Apply placement ───────────────────────────────────────────────────
        if best_pos:
            b_x, b_y, b_z, b_rot, b_dx, b_dy, b_dz, _ = best_pos
        else:
            b_rot = int(solution[idx, 3]) if not can_rotate else 0
            b_dx, b_dy, b_dz = get_rotated_dims(l, w, h, b_rot)
            if az is not None:
                az_x1 = float(az['x1']); az_y1 = float(az['y1'])
                az_x2 = float(az['x2']); az_y2 = float(az['y2'])
                az_z1 = float(az.get('z1', 0))
                az_z2 = float(az.get('z2', wh_hgt))
                # Clamp XY to zone bounds first so we know the actual footprint
                b_x = max(az_x1 + b_dx/2, min(az_x2 - b_dx/2, solution[idx, 0]))
                b_y = max(az_y1 + b_dy/2, min(az_y2 - b_dy/2, solution[idx, 1]))
                # Compute gravity at the SPECIFIC (b_x, b_y) footprint, not zone-wide
                # max.  The old zone-wide max caused overlaps when the tallest item in
                # the zone occupied a different XY region than where this item lands.
                bx1_f = b_x - b_dx / 2;  bx2_f = b_x + b_dx / 2
                by1_f = b_y - b_dy / 2;  by2_f = b_y + b_dy / 2
                fall_gz = max(
                    (p[2] + p[5] for p in placed_items
                     if p[2] < 1000
                     and p[0] + p[3] > bx1_f + 1e-5 and p[0] < bx2_f - 1e-5
                     and p[1] + p[4] > by1_f + 1e-5 and p[1] < by2_f - 1e-5),
                    default=az_z1
                )
                b_z = max(fall_gz, az_z1, float(min_z))
                # Clamp b_z to the zone ceiling so items cannot float above
                # the warehouse.  When the zone is truly full this will cause
                # an overlap — flagged by the final overlap checker — but that
                # is strictly better than physically-impossible items at z>>H.
                max_b_z = az_z2 - b_dz
                if b_z > max_b_z:
                    _log_f.write(
                        f'  [ZONE-FULL-OVERFLOW] item={idx} zone={assigned_zi} '
                        f'raw_z={b_z:.3f} exceeds ceiling ({az_z2:.2f}) - allowing stack to prevent overlap\n'
                    )
                    # b_z = max(max_b_z, az_z1)
            else:
                b_z = max(0.0, float(min_z))
                b_x = max(b_dx/2, min(wh_len - b_dx/2, solution[idx, 0]))
                b_y = max(b_dy/2, min(wh_wid - b_dy/2, solution[idx, 1]))
                # Warehouse-level ceiling clamp (removed to avoid overlap)
                # b_z = min(b_z, max(0.0, wh_hgt - b_dz))

        # ── 3-D overlap resolution ────────────────────────────────────────────
        # After the search/fallback has determined (b_x, b_y, b_z), verify the
        # final AABB does not physically intersect any already-placed item.
        # Any residual overlap (float rounding, zone-tolerance slop) is resolved
        # by pushing b_z up to the conflicting item's top.  This pass is O(N)
        # but only triggers when the search left a small gap uncorrected.
        _ix1, _ix2 = b_x - b_dx / 2, b_x + b_dx / 2
        _iy1, _iy2 = b_y - b_dy / 2, b_y + b_dy / 2
        _OVL_R = 1e-4   # 0.1 mm — reject anything larger as a real overlap
        # Ceiling for this item: zone z2 if assigned, else warehouse height.
        # Prevents overlap-pushing from floating items beyond the warehouse.
        if assigned_zi is not None:
            _z_ceil = float(use_zones[assigned_zi].get('z2', wh_hgt))
        else:
            _z_ceil = wh_hgt
        _resolved = True
        _resolve_iters = 0
        while _resolved and _resolve_iters < 10:
            _resolved = False
            _resolve_iters += 1
            for _px, _py, _pz, _pdx, _pdy, _pdz in placed_items:
                # Fast Z pre-check: skip if no Z overlap with current b_z
                if _pz + _pdz <= b_z + _OVL_R or _pz >= b_z + b_dz - _OVL_R:
                    continue
                # Full XY overlap check
                if (_ix1 < _px + _pdx - _OVL_R and _ix2 > _px + _OVL_R and
                        _iy1 < _py + _pdy - _OVL_R and _iy2 > _py + _OVL_R):
                    # 3-D overlap confirmed — push this item above the conflict,
                    # but never above the zone/warehouse ceiling.  Unresolvable
                    # overlaps are logged by the final overlap checker.
                    new_b_z = _pz + _pdz
                    b_z = new_b_z
                    _resolved = True   # re-check from scratch (b_z changed)
                    break

        solution[idx, 0] = b_x
        solution[idx, 1] = b_y
        solution[idx, 2] = b_z
        solution[idx, 3] = b_rot

        x1_p, y1_p = b_x - b_dx/2, b_y - b_dy/2
        placed_items.append((x1_p, y1_p, b_z, b_dx, b_dy, b_dz))
        global_grid.insert(len(placed_items) - 1,
                           x1_p, y1_p, b_x + b_dx/2, b_y + b_dy/2)
        if _placed_buf is not None:
            _placed_buf[_placed_count, 0] = float(x1_p)
            _placed_buf[_placed_count, 1] = float(y1_p)
            _placed_buf[_placed_count, 2] = float(b_z)
            _placed_buf[_placed_count, 3] = float(b_dx)
            _placed_buf[_placed_count, 4] = float(b_dy)
            _placed_buf[_placed_count, 5] = float(b_dz)
        _placed_count += 1

        if assigned_zi is not None:
            zone_occ.add(assigned_zi, x1_p, y1_p, b_z, b_dx, b_dy, b_dz, is_nf=is_nf)

        _log_item(idx, _current_pass, assigned_zi, is_nf, min_z,
                  b_x, b_y, b_z, b_dx, b_dy, b_dz, used_fallback=not bool(best_pos))

    # ─────────────────────────────────────────────────────────────────────────
    # UNIFIED PLACEMENT PASS: Interleaved NF and Fragile items
    # ─────────────────────────────────────────────────────────────────────────
    # To satisfy all constraints simultaneously (Door proximity, Weight stability,
    # and Fragile-on-NF), we sort ALL assigned items into a single sequence.
    
    # Unified Sort Strategy:
    # ── NF items (fragility=0) ────────────────────────────────────────────────
    #   1. AF bucket DESC  — high-traffic items claim door-proximal spots first.
    #   2. Weight DESC     — heavy NF items form a stable ground layer.
    #   3. AF exact DESC   — tie-break on precise access frequency.
    #   4. Volume DESC     — large items fill gaps before small ones.
    #
    # ── FR items (fragility=1) ────────────────────────────────────────────────
    #   1. Weight DESC     — heavy fragile items form the base of the upper layer.
    #   2. AF bucket DESC  — among equal-weight fragile items, high-AF near door.
    #   3. AF exact DESC   — precise AF tie-break.
    #   4. Weight ASC      — lightweight fragile items placed LAST so gravity
    #                        stacks them on top of heavier fragile items, even
    #                        above other fragile items (lightweight-on-top rule).
    #   5. Volume DESC     — volume tie-break for equal-weight items.
    #
    # All NF items are exhausted before any FR item is processed, guaranteeing
    # zone_occ.max_nf_top is fully populated when FR placement begins.

    # Pre-compute per-item weight percentile rank within fragile group so the
    # lightweight-on-top rule has a stable numeric threshold.
    _fr_weights = np.array([weights[i] for i in indices if fragility[i] == 1], dtype=np.float32)
    _fr_weight_median = float(np.median(_fr_weights)) if len(_fr_weights) > 0 else 0.0

    placement_order = [i for i in indices if item_zone_idx.get(i) is not None]
    placement_order = sorted(placement_order, key=lambda i: (
        fragility[i],                                      # NF (0) always before FR (1)
        # ── NF primary: AF bucket (high-AF to door first) ───────────────────
        -((access_freqs[i] // 2) * 2) if fragility[i] == 0 else 0,
        # ── FR primary: weight bucket (heavy fragile first for stable base) ─
        # Weight divided into coarse buckets so AF can still differentiate
        # items of similar mass.
        -int(weights[i] / 2) * 2 if fragility[i] == 1 else 0,
        # ── NF secondary: exact weight DESC; FR secondary: AF bucket DESC ───
        -weights[i] if fragility[i] == 0 else -((access_freqs[i] // 2) * 2),
        # ── AF exact tie-break for both groups ──────────────────────────────
        -access_freqs[i],
        # ── Lightweight-on-top: within same weight bucket, lighter FR last ──
        # (+weights puts lighter items AFTER heavier ones, forcing them to the
        # top of the fragile stack via gravity.)
        +weights[i] if fragility[i] == 1 else 0,
        # ── Volume tie-break ─────────────────────────────────────────────────
        -volumes[i],
        i
    ))

    _log_f.write(f'--- UNIFIED PASS: {len(placement_order)} items ---\n'); _log_f.flush()
    total_placed = 0
    num_to_place = len(placement_order)

    # Track placed-list index → original solution index for the settle pass.
    _placed_sol_map = []   # _placed_sol_map[k] = solution idx of placed_items[k]
    # Track per-item settle floor (min Z the item is allowed to reach).
    _settle_floor_map = {} # solution_idx -> min_z floor

    for idx in placement_order:
        zi = item_zone_idx.get(idx)
        is_fr = (fragility[idx] == 1)

        if is_fr:
            _current_pass = 'PASS-C(FR)'
            zone_z1 = float(use_zones[zi].get('z1', 0.0))
            zone_z2 = float(use_zones[zi].get('z2', wh_hgt))
            
            # Use the zone's base Z. Gravity and the unified sort order naturally handle stacking:
            # - NF items are placed before FR items, so FR naturally lands on top of NF if they share a column.
            # - Heavy FR items are placed before Light FR items, so Light naturally lands on top of Heavy.
            # Using zone-wide maximums here forced catastrophic towering by artificially marking the entire zone as full.
            min_z_val = zone_z1
            
            _settle_floor_map[idx] = min_z_val
            _place_item(idx, zi, min_z=min_z_val, is_nf=False)
        else:
            _current_pass = 'PASS-A(NF)'
            # Non-Fragile items can start at the very bottom
            _settle_floor_map[idx] = 0.0
            _place_item(idx, zi, min_z=0.0, is_nf=True)

        _placed_sol_map.append(idx)

        total_placed += 1
        if callback and (total_placed % callback_interval == 0 or total_placed == num_to_place):
            callback(solution)

    # ─────────────────────────────────────────────────────────────────────────
    # GRAVITY SETTLE PASS  — compact items downward to eliminate air gaps
    # ─────────────────────────────────────────────────────────────────────────
    # After the initial placement each item sits at the first valid Z found by
    # the search.  Because items are placed in priority/fragility order, a later
    # item may leave a gap below an earlier one.  We close these gaps by pulling
    # every item down to its lowest physically-valid Z (respecting zone floor and
    # fragility constraints), running 3 sweep passes until stable.
    _log_f.write('--- GRAVITY SETTLE PASS ---\n')
    _OVL_S = 1e-4   # 0.1 mm XY overlap threshold used in settle
    n_placed = len(placed_items)

    # Build a settle grid for spatial acceleration of the O(N) inner loop
    _settle_grid = SimpleGrid(wh_len, wh_wid, cell_size=0.5)
    for _sg_k in range(n_placed):
        _sg_x1, _sg_y1, _, _sg_dx, _sg_dy, _ = placed_items[_sg_k]
        _settle_grid.insert(_sg_k, _sg_x1, _sg_y1, _sg_x1 + _sg_dx, _sg_y1 + _sg_dy)

    for _sp in range(3):           # up to 3 passes for multi-item cascades
        _any_moved = False
        # Process items from bottom to top so lower items settle first.
        _settle_order = sorted(range(n_placed), key=lambda k: placed_items[k][2])
        for k in _settle_order:
            sx1, sy1, sz, sdx, sdy, sdz = placed_items[k]
            sol_idx = _placed_sol_map[k]
            zi_s    = item_zone_idx.get(sol_idx, 0)
            zne_s   = use_zones[zi_s]
            z_ceil  = float(zne_s.get('z2', wh_hgt))
            z_floor = _settle_floor_map.get(sol_idx, float(zne_s.get('z1', 0)))

            # Compute lowest valid Z — conflict-resolution loop avoids clipping.
            # Start at z_floor and push new_z above any 3D conflict at the
            # candidate position, repeating until no conflicts remain.
            new_z = z_floor
            sx2, sy2 = sx1 + sdx, sy1 + sdy
            # Use spatial grid to only check nearby items (O(k) vs O(N))
            _nearby = _settle_grid.query(sx1, sy1, sx2, sy2)
            _s_resolved = True
            _s_iters = 0
            while _s_resolved and _s_iters < 10:
                _s_resolved = False
                _s_iters += 1
                for j in _nearby:
                    if j == k:
                        continue
                    px, py, pz, pdx, pdy, pdz = placed_items[j]
                    # Z overlap at candidate new_z?
                    if pz + pdz <= new_z + _OVL_S or pz >= new_z + sdz - _OVL_S:
                        continue
                    # XY footprint overlap?
                    if (sx1 < px + pdx - _OVL_S and sx2 > px + _OVL_S and
                            sy1 < py + pdy - _OVL_S and sy2 > py + _OVL_S):
                        new_z = pz + pdz   # push above conflict
                        _s_resolved = True
                        break

            # Lower item only if it genuinely descends (ignoring ceiling boundary if already exceeded)
            if new_z < sz - _OVL_S:
                placed_items[k] = (sx1, sy1, new_z, sdx, sdy, sdz)
                solution[sol_idx, 2] = new_z   # base Z stored in solution
                _any_moved = True
        if not _any_moved:
            break   # stable — stop early

    _log_f.write(f'  Settle done in {_sp + 1} pass(es). Items moved: {_any_moved}\n')

    # ─────────────────────────────────────────────────────────────────────────
    # OVERLAP CHECKER — detect, log, and resolve any residual 3D AABB overlaps
    # ─────────────────────────────────────────────────────────────────────────
    # After settle, verify no two items physically intersect. Any overlap larger
    # than _OVL_CHK is logged with both item IDs, XYZ positions, and overlap
    # volume, then repaired by pushing the upper item above the lower.
    _log_f.write('--- OVERLAP CHECK ---\n')
    _OVL_CHK = 1e-4   # 0.1 mm tolerance — anything beyond is a real overlap
    _overlap_count = 0
    _overlap_vol_total = 0.0
    # Rebuild settle grid after settle pass (positions may have changed)
    _ovl_grid = SimpleGrid(wh_len, wh_wid, cell_size=0.5)
    for _og_k in range(len(placed_items)):
        _og_x1, _og_y1, _, _og_dx, _og_dy, _ = placed_items[_og_k]
        _ovl_grid.insert(_og_k, _og_x1, _og_y1, _og_x1 + _og_dx, _og_y1 + _og_dy)
    _chk_resolved = True
    _chk_passes = 0
    while _chk_resolved and _chk_passes < 5:
        _chk_resolved = False
        _chk_passes += 1
        for a in range(len(placed_items)):
            ax, ay, az_, adx, ady, adz = placed_items[a]
            ax2, ay2, az2 = ax + adx, ay + ady, az_ + adz
            _ovl_nearby = _ovl_grid.query(ax, ay, ax2, ay2)
            for b in _ovl_nearby:
                if b <= a:
                    continue
                bx, by, bz_, bdx, bdy, bdz = placed_items[b]
                bx2, by2, bz2 = bx + bdx, by + bdy, bz_ + bdz
                # 3D AABB intersection (with tolerance)
                ox = min(ax2, bx2) - max(ax, bx)
                oy = min(ay2, by2) - max(ay, by)
                oz = min(az2, bz2) - max(az_, bz_)
                if ox > _OVL_CHK and oy > _OVL_CHK and oz > _OVL_CHK:
                    _overlap_count += 1
                    _ovl_vol = ox * oy * oz
                    _overlap_vol_total += _ovl_vol
                    sol_a = _placed_sol_map[a]
                    sol_b = _placed_sol_map[b]
                    _log_f.write(
                        f'  [OVERLAP] item_A={sol_a} item_B={sol_b} '
                        f'A_xyz=({ax:.3f},{ay:.3f},{az_:.3f}) A_sz=({adx:.2f}x{ady:.2f}x{adz:.2f}) '
                        f'B_xyz=({bx:.3f},{by:.3f},{bz_:.3f}) B_sz=({bdx:.2f}x{bdy:.2f}x{bdz:.2f}) '
                        f'overlap=({ox:.4f}x{oy:.4f}x{oz:.4f}) vol={_ovl_vol:.5f}m^3\n'
                    )
                
                # --- STACKING CHECK (Fragile Under Non-Fragile) ---
                # Detect and log violations where a fragile item is under a non-fragile item in the same column.
                # Threshold for XY overlap to consider it "stacked" is 5cm.
                if ox > 0.05 and oy > 0.05:
                    sol_a, sol_b = _placed_sol_map[a], _placed_sol_map[b]
                    frag_a, frag_b = fragility[sol_a], fragility[sol_b]
                    # Check A under B
                    if az2 <= bz_ + _OVL_CHK:
                        if frag_a == 1 and frag_b == 0:
                             _log_f.write(f'  [STACKING ERROR] Fragile item {sol_a} is underneath Non-Fragile item {sol_b}!\n')
                    # Check B under A
                    elif bz2 <= az_ + _OVL_CHK:
                        if frag_b == 1 and frag_a == 0:
                             _log_f.write(f'  [STACKING ERROR] Fragile item {sol_b} is underneath Non-Fragile item {sol_a}!\n')
                    # Repair: push the upper item above the lower (by Z base)
                    if az_ >= bz_:
                        upper_k, lower_k = a, b
                        new_upper_z = bz_ + bdz
                    else:
                        upper_k, lower_k = b, a
                        new_upper_z = az_ + adz
                    ux, uy, uz, udx, udy, udz = placed_items[upper_k]
                    sol_u = _placed_sol_map[upper_k]
                    zi_u = item_zone_idx.get(sol_u, 0)
                    z_ceil_u = float(use_zones[zi_u].get('z2', wh_hgt)) if zi_u is not None else wh_hgt
                    if new_upper_z + udz <= z_ceil_u + _OVL_CHK:
                        placed_items[upper_k] = (ux, uy, new_upper_z, udx, udy, udz)
                        solution[sol_u, 2] = new_upper_z
                        _chk_resolved = True
                        break
            if _chk_resolved:
                break
    if _overlap_count == 0:
        _log_f.write('  [OK] No overlaps detected — all items are physically separated.\n')
    else:
        _log_f.write(
            f'  [SUMMARY] overlaps_found={_overlap_count}  '
            f'total_overlap_vol={_overlap_vol_total:.4f}m^3  '
            f'resolution_passes={_chk_passes}\n'
        )
    _log_f.flush()

    # ── Final Item Placement Logging and Strict Overlap Check ─────────────────
    _log_f.write('--- FINAL PLACEMENT LOGGING ---\n')
    _final_overlap_count = 0
    # Build spatial grid for final overlap check to avoid O(N^2)
    _final_grid = SimpleGrid(wh_len, wh_wid, cell_size=0.5)
    _final_bounds = []  # (x1, y1, z1, x2, y2, z2) per item
    for i in range(num_items):
        l, w, h = items_props[i, 0:3]
        cx, cy, cz, rot = solution[i, 0:4]
        dx, dy, dz = get_rotated_dims(l, w, h, int(rot))
        x1_i, y1_i, z1_i = cx - dx / 2, cy - dy / 2, cz
        x2_i, y2_i, z2_i = cx + dx / 2, cy + dy / 2, cz + dz
        _final_bounds.append((x1_i, y1_i, z1_i, x2_i, y2_i, z2_i, dx, dy, dz))
        _final_grid.insert(i, x1_i, y1_i, x2_i, y2_i)
    for i in range(num_items):
        name_str = f"({item_names[i]})" if item_names and i < len(item_names) else f"Item {i}"
        cx, cy, cz = solution[i, 0], solution[i, 1], solution[i, 2]
        x1_i, y1_i, z1_i, x2_i, y2_i, z2_i, dx, dy, dz = _final_bounds[i]

        _log_f.write(f'  {name_str}: Pos=({cx:.3f}, {cy:.3f}, {cz:.3f}) Size=({dx:.2f}x{dy:.2f}x{dz:.2f}) '
                     f'Bounds=[{x1_i:.3f}, {x2_i:.3f}], [{y1_i:.3f}, {y2_i:.3f}], [{z1_i:.3f}, {z2_i:.3f}]\n')

        _final_nearby = _final_grid.query(x1_i, y1_i, x2_i, y2_i)
        for j in _final_nearby:
            if j <= i:
                continue
            x1_j, y1_j, z1_j, x2_j, y2_j, z2_j, _, _, _ = _final_bounds[j]

            ox = min(x2_i, x2_j) - max(x1_i, x1_j)
            oy = min(y2_i, y2_j) - max(y1_i, y1_j)
            oz = min(z2_i, z2_j) - max(z1_i, z1_j)

            if ox > 1e-4 and oy > 1e-4 and oz > 1e-4:
                j_name_str = f"({item_names[j]})" if item_names and j < len(item_names) else f"Item {j}"
                if (x1_i >= x1_j - 1e-4 and x2_i <= x2_j + 1e-4 and 
                    y1_i >= y1_j - 1e-4 and y2_i <= y2_j + 1e-4 and 
                    z1_i >= z1_j - 1e-4 and z2_i <= z2_j + 1e-4):
                    _log_f.write(f'  [CRITICAL ERROR] {name_str} is completely INSIDE {j_name_str}!\n')
                elif (x1_j >= x1_i - 1e-4 and x2_j <= x2_i + 1e-4 and 
                      y1_j >= y1_i - 1e-4 and y2_j <= y2_i + 1e-4 and 
                      z1_j >= z1_i - 1e-4 and z2_j <= z2_i + 1e-4):
                    _log_f.write(f'  [CRITICAL ERROR] {j_name_str} is completely INSIDE {name_str}!\n')
                else:
                    _log_f.write(f'  [ERROR] {name_str} intersects with {j_name_str} by ({ox:.3f}x{oy:.3f}x{oz:.3f})\n')
                _final_overlap_count += 1

    if _final_overlap_count == 0:
        _log_f.write('  [FINAL CHECK OK] No items are inside each other or overlapping in final placement.\n')
    else:
        _log_f.write(f'  [FINAL CHECK FAILED] {_final_overlap_count} overlaps/inside errors found in final placement!\n')
    _log_f.flush()

    # ── Final zone summary ────────────────────────────────────────────────────
    _log_f.write('--- ZONE SUMMARY ---\n')
    for zi in range(num_zones):
        items_in_zone = zone_occ.items[zi]
        nf_in  = [it for it in items_in_zone if True]   # all items
        nf_cnt = sum(1 for it in items_in_zone
                     if it[2] < zone_occ.max_nf_top[zi] - 0.001 or
                        abs(it[2] - use_zones[zi].get('z1', 0)) < 0.01)
        fr_cnt = len(items_in_zone) - nf_cnt
        z_vals = [it[2] for it in items_in_zone] if items_in_zone else [0]
        _log_f.write(
            f'  Zone {zi}: total={len(items_in_zone)} '
            f'nf_top={round(zone_occ.max_nf_top[zi],3)} '
            f'fr_top={round(zone_occ.max_fr_top[zi],3)} '
            f'z_min={round(min(z_vals),3)} '
            f'z_max={round(max(z_vals),3)}\n'
        )
    _log_f.write('--- END ---\n')
    _log_f.flush()
    _log_f.close()

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


def _rotated_dims_batch(n, l_arr, w_arr, h_arr, rot_mod):
    """Vectorized rotation: returns (cur_l, cur_w, cur_h) for all 6 axis codes."""
    cur_l = np.empty(n, dtype=np.float32)
    cur_w = np.empty(n, dtype=np.float32)
    cur_h = np.empty(n, dtype=np.float32)
    _rots = [
        (l_arr, w_arr, h_arr),
        (w_arr, l_arr, h_arr),
        (l_arr, h_arr, w_arr),
        (h_arr, l_arr, w_arr),
        (w_arr, h_arr, l_arr),
        (h_arr, w_arr, l_arr),
    ]
    for code, (rl, rw, rh) in enumerate(_rots):
        m = rot_mod == code
        if m.any():
            cur_l[m] = rl[m]
            cur_w[m] = rw[m]
            cur_h[m] = rh[m]
    return cur_l, cur_w, cur_h


def fitness_function_numpy(solution, items_props=None, warehouse_dims=None, weights=None, valid_z=None, exclusion_zones_arr=None):
    if items_props is None: items_props = _pool_items_props
    if warehouse_dims is None: warehouse_dims = _pool_wh_dims
    if valid_z is None: valid_z = _pool_valid_z
    if exclusion_zones_arr is None: exclusion_zones_arr = _pool_exclusion_zones

    solution = solution.astype(np.float32, copy=False)

    grouping = 0.0
    wh_vol = warehouse_dims[0] * warehouse_dims[1] * warehouse_dims[2]
    space_util = np.sum(items_props[:, 0] * items_props[:, 1] * items_props[:, 2]) / wh_vol if wh_vol > 0 else 0

    door_x = warehouse_dims[3] if len(warehouse_dims) >= 5 else 0
    door_y = warehouse_dims[4] if len(warehouse_dims) >= 5 else 0
    access_scores = 1.0 / (1.0 + np.sqrt((solution[:, 0] - door_x) ** 2 + (solution[:, 1] - door_y) ** 2))
    freqs = items_props[:, 5]
    accessibility = np.average(access_scores, weights=freqs) if np.sum(freqs) > 1e-9 else np.mean(access_scores)
    
    # Rotated dimensions (computed once, reused for stability / overlap / stackability)
    n = len(solution)
    l_arr = items_props[:, 0]
    w_arr = items_props[:, 1]
    h_arr = items_props[:, 2]
    rots = solution[:, 3].astype(int)
    rot_mod = rots % 6
    cur_l, cur_w, cur_h = _rotated_dims_batch(n, l_arr, w_arr, h_arr, rot_mod)

    x = solution[:, 0]
    y = solution[:, 1]
    z = solution[:, 2]

    # Stability: floor support + item-on-item support
    is_stable = solution[:, 2] <= 0.01
    for i in np.where(~is_stable)[0]:
        below = np.where(z + cur_h < z[i] + 0.05)[0]
        if not len(below):
            continue
        supporters = below[np.abs((z[below] + cur_h[below]) - z[i]) < 0.05]
        if not len(supporters):
            continue
        ix1, ix2 = x[i] - cur_l[i] / 2, x[i] + cur_l[i] / 2
        iy1, iy2 = y[i] - cur_w[i] / 2, y[i] + cur_w[i] / 2
        ox = np.maximum(0, np.minimum(ix2, x[supporters] + cur_l[supporters] / 2) -
                           np.maximum(ix1, x[supporters] - cur_l[supporters] / 2))
        oy = np.maximum(0, np.minimum(iy2, y[supporters] + cur_w[supporters] / 2) -
                           np.maximum(iy1, y[supporters] - cur_w[supporters] / 2))
        if np.any(ox * oy > cur_l[i] * cur_w[i] * 0.2):
            is_stable[i] = True

    stability = np.mean(is_stable)
    
    zone_penalty = 0
    if exclusion_zones_arr is not None and len(exclusion_zones_arr) > 0:
        ix = solution[:, 0:1]
        iy = solution[:, 1:2]
        radii = (np.maximum(items_props[:, 0], items_props[:, 1]) / 2.0).reshape(-1, 1)
        z_cx = (exclusion_zones_arr[:, 0] + exclusion_zones_arr[:, 2]) / 2
        z_cy = (exclusion_zones_arr[:, 1] + exclusion_zones_arr[:, 3]) / 2
        z_hw = (exclusion_zones_arr[:, 2] - exclusion_zones_arr[:, 0]) / 2
        z_hh = (exclusion_zones_arr[:, 3] - exclusion_zones_arr[:, 1]) / 2
        collisions = (np.abs(ix - z_cx) < radii + z_hw) & (np.abs(iy - z_cy) < radii + z_hh)
        zone_penalty = np.sum(collisions) / n
        
    # --- Item-Item Overlap (batched to bound memory) ---
    overlap_count = 0
    if n > 0:
        hw = cur_l / 2
        hh = cur_w / 2
        z1 = z
        z2 = z + cur_h

        BATCH_SIZE = 512
        
        for i_start in range(0, n, BATCH_SIZE):
            i_end = min(i_start + BATCH_SIZE, n)
            x_batch  = x[i_start:i_end].reshape(-1, 1)
            y_batch  = y[i_start:i_end].reshape(-1, 1)
            z1_batch = z1[i_start:i_end].reshape(-1, 1)
            z2_batch = z2[i_start:i_end].reshape(-1, 1)
            hw_batch = hw[i_start:i_end].reshape(-1, 1)
            hh_batch = hh[i_start:i_end].reshape(-1, 1)
            for j_start in range(0, n, BATCH_SIZE):
                j_end = min(j_start + BATCH_SIZE, n)
                x_other = x[j_start:j_end].reshape(1, -1)
                y_other = y[j_start:j_end].reshape(1, -1)
                z1_other = z1[j_start:j_end].reshape(1, -1)
                z2_other = z2[j_start:j_end].reshape(1, -1)
                hw_other = hw[j_start:j_end].reshape(1, -1)
                hh_other = hh[j_start:j_end].reshape(1, -1)
                
                overlap_x = np.abs(x_batch - x_other) < (hw_batch + hw_other - 0.01)
                overlap_y = np.abs(y_batch - y_other) < (hh_batch + hh_other - 0.01)
                overlap_z = (z2_batch > z1_other + 0.01) & (z1_batch < z2_other - 0.01)
                overlap_count += np.sum(overlap_x & overlap_y & overlap_z)

        # Subtract self-overlaps and halve for symmetry
        overlap_count = (overlap_count - n) / 2.0
        overlap_penalty = max(0.0, overlap_count) / n

    # --- Stackability Enforcement ---
    stackability_penalty = 0
    if n > 1:
        stackable = items_props[:, 4]
        z_tops = z + cur_h
        hw_s = cur_l / 2
        hh_s = cur_w / 2

        violations = 0
        BATCH_SIZE_STACK = 128
        for i_start in range(0, n, BATCH_SIZE_STACK):
            i_end = min(i_start + BATCH_SIZE_STACK, n)
            z_i   = z[i_start:i_end].reshape(-1, 1)
            x_i   = x[i_start:i_end].reshape(-1, 1)
            y_i   = y[i_start:i_end].reshape(-1, 1)
            hw_i  = hw_s[i_start:i_end].reshape(-1, 1)
            hh_i  = hh_s[i_start:i_end].reshape(-1, 1)
            for j_start in range(0, n, BATCH_SIZE_STACK):
                j_end = min(j_start + BATCH_SIZE_STACK, n)
                resting = np.abs(z_i - z_tops[j_start:j_end].reshape(1, -1)) < 0.1
                if not np.any(resting):
                    continue
                dx = np.abs(x_i - x[j_start:j_end].reshape(1, -1))
                dy = np.abs(y_i - y[j_start:j_end].reshape(1, -1))
                hw_j = hw_s[j_start:j_end].reshape(1, -1)
                hh_j = hh_s[j_start:j_end].reshape(1, -1)
                is_supported = resting & (dx < (hw_i + hw_j) * 0.5) & (dy < (hh_i + hh_j) * 0.5)
                if np.any(is_supported):
                    bad = is_supported & (stackable[j_start:j_end].reshape(1, -1) < 0.5)
                    violations += np.sum(np.any(bad, axis=1))

        stackability_penalty = violations / n
    
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
            grouping = 1.0 / (1.0 + (total_dist_sum / count) * 0.1)
        else:
            grouping = 1.0
    else:
        grouping = 0

    total_weight = sum(weights.values())
    norm_weights = {k: v / total_weight for k, v in weights.items()} if total_weight > 1e-9 else weights

    fitness = (norm_weights.get('space', 0) * space_util +
               norm_weights.get('accessibility', 0) * accessibility +
               norm_weights.get('stability', 0) * stability +
               norm_weights.get('grouping', 0) * grouping)

    for penalty in (zone_penalty, overlap_penalty, stackability_penalty):
        if penalty > 0:
            fitness *= 1.0 / (1.0 + penalty * 5.0)

    if fitness <= 1e-6 and random.random() < 0.001:
        with open('thread_debug.log', 'a') as f:
            f.write(f"Zero Fit: Overlap={overlap_penalty:.4f}, Zone={zone_penalty:.4f}, Stack={stackability_penalty:.4f}\n")

    wh_hgt_val = warehouse_dims[2]
    if wh_hgt_val > 0:
        fitness *= 1.0 - (np.mean(solution[:, 2]) / wh_hgt_val) * 0.15

    return fitness, space_util, accessibility, stability, grouping


def fitness_function(solution_list, items, warehouse, weights=None):
    num_items = len(items)
    item_map = {item['id']: i for i, item in enumerate(items)}

    sol_array = np.zeros((num_items, 4))
    for item_sol in solution_list:
        idx = item_map.get(item_sol['id'])
        if idx is not None:
            sol_array[idx] = [item_sol['x'], item_sol['y'], item_sol['z'], item_sol['rotation']]

    items_props = np.zeros((num_items, 10))
    for i, item in enumerate(items):
        items_props[i] = [
            item['length'], item['width'], item['height'],
            item['can_rotate'], item['stackable'],
            item['access_freq'], item.get('weight', 0),
            hash(item.get('category', '')) % 10000,
            item.get('fragility', 0),
            item.get('priority', 1)
        ]

    wh_dims = (warehouse['length'], warehouse['width'], warehouse['height'],
               warehouse.get('door_x', 0), warehouse.get('door_y', 0))
    valid_z = get_valid_z_positions(warehouse)

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
