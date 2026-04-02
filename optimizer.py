import math
import random
import numpy as np
import multiprocessing
from database import get_exclusion_zones

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
                            target_x, target_y, is_fast, device):
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

            ov_x = (mx_e > px_e + 0.001) & (cx_e < px_e + pdx_e - 0.001)
            ov_y = (my_e > py_e + 0.001) & (cy_e < py_e + pdy_e - 0.001)
            overlaps = ov_x & ov_y & valid_sup.unsqueeze(0)          # (N_t, P)

            z_tops_e = z_tops_all.unsqueeze(0).expand(N_t, -1)
            gz = torch.where(overlaps, z_tops_e, torch.zeros_like(z_tops_e))
            gravity_z_t = gz.max(dim=1).values                        # (N_t,)
        else:
            gravity_z_t = torch.zeros(N_t, dtype=torch.float32, device=device)

        # --- Zone validation (restricted to assigned zone when active) ---
        final_z_t    = torch.full((N_t,), float('inf'), dtype=torch.float32, device=device)
        valid_zone_t = torch.zeros(N_t, dtype=torch.bool, device=device)

        for zne in valid_zones:
            zx1, zx2 = float(zne['x1']), float(zne['x2'])
            zy1, zy2 = float(zne['y1']), float(zne['y2'])
            zz1 = float(zne.get('z1', 0))
            zz2 = float(zne.get('z2', wh_hgt))

            in_zone = (
                (all_cx >= zx1 - 0.01) & (max_x_t <= zx2 + 0.01) &
                (all_cy >= zy1 - 0.01) & (max_y_t <= zy2 + 0.01) &
                valid_bounds
            )
            pz_z    = torch.clamp(gravity_z_t, min=zz1)
            fits    = (pz_z + dz_f) <= (zz2 + 0.001)
            zone_ok = in_zone & fits
            update  = zone_ok & (pz_z < final_z_t)
            final_z_t    = torch.where(update, pz_z, final_z_t)
            valid_zone_t = valid_zone_t | zone_ok

        valid_t = valid_zone_t & valid_bounds
        if not valid_t.any():
            continue

        center_x_t = all_cx + dx_f / 2
        center_y_t = all_cy + dy_f / 2
        dist_t = (center_x_t - target_x) ** 2 + (center_y_t - target_y) ** 2

        # Zone-relative coords for compact front-left packing within each zone.
        # When no zone is assigned (single-zone / fallback) this reduces to (cy, cx).
        rel_y_t = all_cy - zone_oy
        rel_x_t = all_cx - zone_ox

        # Fast-mode early exit: floor placement near zone origin
        if is_fast:
            # Aggressive Early Stop: If we find a floor placement near the predicted target, 
            # take it immediately without evaluating further candidates or rotations.
            fc = valid_t & (final_z_t <= 0.01) & (dist_t < 0.1)
            if fc.any():
                i = fc.nonzero(as_tuple=True)[0][0].item()
                sc = (final_z_t[i].item(), rel_y_t[i].item(),
                      rel_x_t[i].item(), dist_t[i].item())
                return (center_x_t[i].item(), center_y_t[i].item(),
                        final_z_t[i].item(), rot_code, dx, dy, dz, sc)

        # Lexicographic argmin: (floor_bias, final_z, zone-rel-y, zone-rel-x, dist_to_target)
        # floor_bias: 0 if at floor, 1 if stacked. This forces ground-level packing first.
        fz = torch.where(valid_t, final_z_t, BIG)
        floor_bias = torch.where(fz <= 0.01, torch.zeros_like(fz), torch.ones_like(fz))
        min_fb = floor_bias.min()
        m = valid_t & (floor_bias <= min_fb + 1e-6)

        fz_m = torch.where(m, fz, BIG)
        min_fz = fz_m.min()
        m = m & (fz_m <= min_fz + 1e-6)

        ry = torch.where(m, rel_y_t, BIG); min_ry = ry.min(); m = m & (ry <= min_ry + 1e-6)
        rx = torch.where(m, rel_x_t, BIG); min_rx = rx.min(); m = m & (rx <= min_rx + 1e-6)
        dt = torch.where(m, dist_t,  BIG)
        bi = dt.argmin().item()

        if valid_t[bi]:
            sc   = (floor_bias[bi].item(), final_z_t[bi].item(), rel_y_t[bi].item(),
                    rel_x_t[bi].item(), dist_t[bi].item())
            cand = (center_x_t[bi].item(), center_y_t[bi].item(),
                    final_z_t[bi].item(), rot_code, dx, dy, dz, sc)
            if best is None or sc < best[7]:
                best = cand

    return best


def repair_solution_compact(solution, items_props, warehouse_dims, allocation_zones=None, layer_heights=None, callback=None, callback_interval=50, fast_mode=False, max_candidates=None):
    """Repair solution by placing items in valid positions with gravity."""
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
    use_zones = allocation_zones if allocation_zones else [{'x1':0, 'y1':0, 'x2':wh_len, 'y2':wh_wid, 'z1':0, 'z2':wh_hgt}]
    
    # Sort zones by Z-base (Bottom first), then Y1, then X1 for back-to-front, bottom-to-top filling
    use_zones = sorted(use_zones, key=lambda zn: (zn.get('z1', 0), zn.get('y1', 0), zn.get('x1', 0)))
    
    num_zones = len(use_zones)

    if num_zones >= 2:
        # Separate fragile vs non-fragile, each sorted heavy-first for stability
        fragile_indices = [i for i in indices if fragility[i] == 1]
        non_fragile_indices = [i for i in indices if fragility[i] != 1]
        
        fragile_list = sorted(fragile_indices, key=lambda i: (-weights[i], -volumes[i], i))
        non_fragile_list = sorted(non_fragile_indices, key=lambda i: (-weights[i], -volumes[i], i))

        # --- Look-Ahead Capacity-Aware Multi-Zone Assignment ---
        #
        # Strategy:
        #   1. Simulate NF assignment (dry-run) to predict which upper zones will
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
            l, w, h      = items_props[item_i, 0:3]
            eligible     = []
            for zi in target_zones:
                zne   = use_zones[zi]
                max_d = max(l, w, h)
                z_max = max(zne['x2'] - zne['x1'],
                            zne['y2'] - zne['y1'],
                            zne['z2'] - zne['z1'])
                if max_d <= z_max + 0.1:
                    cap     = zone_caps[zi]
                    rem_pct = (cap - used_vols[zi]) / cap if cap > 0 else 0
                    eligible.append((zi, rem_pct))
            if eligible:
                random.shuffle(eligible)          # break ties randomly
                return max(eligible, key=lambda x: x[1])[0]
            # Fallback: round-robin within level
            return target_zones[item_i % len(target_zones)]

        # ── PHASE 1: Simulate NF assignment (dry-run) to predict fragile overflow ──
        sim_used    = {zi: 0.0 for zi in range(num_zones)}
        sim_l_idx   = 0
        for item_i in non_fragile_list:
            sim_l_idx          = _overflow_level(sim_used, sim_l_idx)
            best_zi            = _best_zone_in_level(item_i, sim_l_idx, sim_used)
            sim_used[best_zi] += volumes[item_i]

        # Simulate fragile assignment on top of the NF simulation.
        sim_frag_l_idx              = sim_l_idx
        fragile_overflow_per_upper  = {}   # upper_zi -> [item indices]
        for item_i in fragile_list:
            sim_frag_l_idx              = _overflow_level(sim_used, sim_frag_l_idx)
            best_zi                     = _best_zone_in_level(item_i, sim_frag_l_idx, sim_used)
            sim_used[best_zi]          += volumes[item_i]
            if sim_frag_l_idx > 0:        # item lands in an upper zone
                fragile_overflow_per_upper.setdefault(best_zi, []).append(item_i)

        # ── PHASE 2: Pre-reserve NF base layer for upper zones with fragile overflow ──
        nf_reserved_for_upper = set()    # item indices already assigned to an upper zone

        for upper_zi, frag_items in sorted(fragile_overflow_per_upper.items()):
            if not frag_items:
                continue

            upper_floor_area = zone_floor_areas[upper_zi]
            upper_cap        = zone_caps[upper_zi]

            # Target floor coverage: match fragile footprint, capped at 40% of zone floor.
            fragile_footprint = sum(
                items_props[i, 0] * items_props[i, 1] for i in frag_items
            )
            target_coverage = min(fragile_footprint, upper_floor_area * 0.40)

            # Each zone is an INDEPENDENT stacking column: NF items sit at the zone floor,
            # fragile items stack on top of NF. They do NOT compete for the same volume —
            # they occupy different height slices within the same zone.
            # Therefore the NF budget is purely 40% of the zone's own capacity
            # (roughly the lower 40% of zone height), irrespective of fragile item sizes.
            available_for_nf = upper_cap * 0.40
            # Safety: skip only if there is genuinely no floor area to place NF.
            if upper_floor_area <= 0 or target_coverage <= 0:
                continue

            # Pick smallest-volume NF items first to minimise disruption to bottom zone.
            nf_candidates = sorted(
                [i for i in non_fragile_list if i not in nf_reserved_for_upper],
                key=lambda i: volumes[i]
            )

            moved_coverage = 0.0
            moved_vol      = 0.0
            deferred       = 0
            for item_i in nf_candidates:
                if moved_coverage >= target_coverage or moved_vol >= available_for_nf:
                    break
                item_zone_idx[item_i]        = upper_zi
                zone_used_vols[upper_zi]    += volumes[item_i]
                nf_reserved_for_upper.add(item_i)
                moved_coverage              += items_props[item_i, 0] * items_props[item_i, 1]
                moved_vol                   += volumes[item_i]
                deferred                    += 1

            print(
                f"DEBUG look-ahead: Pre-reserved {deferred} NF items → upper zone {upper_zi} "
                f"(coverage={moved_coverage:.2f}/{target_coverage:.2f}, "
                f"vol={moved_vol:.2f}/{available_for_nf:.2f})"
            )

        # ── PHASE 3: Fill bottom zone with remaining NF items (maximise bottom shelf) ──
        nf_for_bottom = [i for i in non_fragile_list if i not in nf_reserved_for_upper]
        bottom_l_idx  = 0
        for item_i in nf_for_bottom:
            bottom_l_idx              = _overflow_level(zone_used_vols, bottom_l_idx)
            best_zi                   = _best_zone_in_level(item_i, bottom_l_idx, zone_used_vols)
            item_zone_idx[item_i]     = best_zi
            zone_used_vols[best_zi]  += volumes[item_i]

        # ── PHASE 4: Assign fragile items (overflow lands on top of NF base) ──
        frag_l_idx = bottom_l_idx
        for item_i in fragile_list:
            frag_l_idx                = _overflow_level(zone_used_vols, frag_l_idx)
            best_zi                   = _best_zone_in_level(item_i, frag_l_idx, zone_used_vols)
            item_zone_idx[item_i]     = best_zi
            zone_used_vols[best_zi]  += volumes[item_i]

        # ── PHASE 5: Build stacking order: NF-first (all zones), then F-last (all zones) ──
        # Guarantees that within every zone the non-fragile base layer is placed before
        # any fragile items are dropped on top — regardless of which level the zone is on.
        sorted_indices = []
        for z_idx in range(num_zones):
            sorted_indices.extend([i for i in non_fragile_list if item_zone_idx.get(i) == z_idx])
        for z_idx in range(num_zones):
            sorted_indices.extend([i for i in fragile_list     if item_zone_idx.get(i) == z_idx])

        print(
            f"DEBUG look-ahead: {len(nf_reserved_for_upper)} NF items pre-reserved for upper zones, "
            f"{len(nf_for_bottom)} NF items assigned to bottom zone, "
            f"across {len(unique_z_levels)} Z-levels."
        )




    else:
        # Single zone: keep original global sort (backward compatible)
        sorted_indices = sorted(indices, key=lambda i: (fragility[i], -weights[i], -volumes[i], i))
        item_zone_idx = None

    # Tracking placed items: (x, y, z, dx, dy, dz)
    placed_items = []
    # Spatial Grid for O(1) neighboring item lookup
    grid = SimpleGrid(wh_len, wh_wid, cell_size=4.0)

    # GPU/vectorized acceleration: pre-allocate placed-items buffer
    if _TORCH_AVAILABLE:
        _dev = _TORCH_DEVICE
        _placed_buf = _torch.zeros((num_items, 6), dtype=_torch.float32, device=_dev)
        _placed_count = 0
    else:
        _dev = None
        _placed_buf = None
        _placed_count = 0

    for idx in sorted_indices:
        l, w, h = items_props[idx, 0:3]
        can_rotate = int(items_props[idx, 3])
        
        # Try all 6 rotations; fragile items have can_rotate=0 so they're unaffected
        rots = [0, 1, 2, 3, 4, 5] if can_rotate else [int(solution[idx, 3])]
        
        best_pos = None
        candidates = set()

        target_x = solution[idx, 0]
        target_y = solution[idx, 1]

        # Resolve assigned zone once for this item
        _az_idx = item_zone_idx.get(idx) if item_zone_idx is not None else None
        assigned_zone = use_zones[_az_idx] if _az_idx is not None else None

        # Warm-start: predicted position
        candidates.add((target_x - l/2, target_y - w/2))
        candidates.add((target_x - w/2, target_y - l/2))

        # Zone lattice grid + origin
        if assigned_zone is not None:
            candidates.add((assigned_zone['x1'], assigned_zone['y1']))
            _gx = assigned_zone['x1']
            while _gx < assigned_zone['x2'] - 0.01:
                _gy = assigned_zone['y1']
                while _gy < assigned_zone['y2'] - 0.01:
                    candidates.add((_gx, _gy))
                    _gy += 0.5
                _gx += 0.5
            lattice_zones = [assigned_zone]
        else:
            lattice_zones = use_zones

        for zne in lattice_zones:
            candidates.add((zne['x1'], zne['y1']))

        # Touch-points from recent placed items for tight adjacency
        for (px, py, pz, pdx, pdy, pdz) in placed_items[-100:]:
            candidates.add((px + pdx, py))
            candidates.add((px, py + pdy))
            candidates.add((px, py))

        valid_candidates = [(cx, cy) for (cx, cy) in candidates
                            if 0 <= cx < wh_len and 0 <= cy < wh_wid]

        if assigned_zone is not None:
            sort_tx = max(assigned_zone['x1'], min(assigned_zone['x2'] - l, target_x - l/2))
            sort_ty = max(assigned_zone['y1'], min(assigned_zone['y2'] - w, target_y - w/2))
        else:
            sort_tx, sort_ty = target_x - l/2, target_y - w/2

        sorted_candidates = sorted(valid_candidates, key=lambda p: (
            (p[0] - sort_tx)**2 + (p[1] - sort_ty)**2
        ))
        
        # Determine candidate limit based on fast_mode and user override
        if max_candidates is not None:
            cand_limit = max_candidates
        else:
            # Force deep search for assigned zones even in fast mode to prevent
            # "blindness" to remote shelves (like Shelf C) during ML inference.
            if assigned_zone is not None or (not fast_mode and idx < len(volumes) and volumes[idx] > 0.5):
                cand_limit = 500
            else:
                cand_limit = 25 if fast_mode else 150 

            
        search_candidates = sorted_candidates[:cand_limit]
        
        def perform_search(candidates_to_check, is_fast, check_zones=None):
            zones = check_zones if check_zones is not None else use_zones
            zone_ox_cpu = float(zones[0]['x1']) if zones else 0.0
            zone_oy_cpu = float(zones[0]['y1']) if zones else 0.0
            best = None
            for rot in rots:
                dx, dy, dz = get_rotated_dims(l, w, h, rot)
                for (cx, cy) in candidates_to_check:
                    max_x, max_y = cx + dx, cy + dy
                    if max_x > wh_len + 0.001 or max_y > wh_wid + 0.001:
                        continue
                    gravity_z = 0.0
                    for p_idx in grid.query(cx, cy, max_x, max_y):
                        px, py, pz, pdx, pdy, pdz = placed_items[p_idx]
                        if pz < 1000 and max_x > px + 0.001 and cx < px + pdx - 0.001 and \
                                max_y > py + 0.001 and cy < py + pdy - 0.001:
                            top_z = pz + pdz
                            if top_z > gravity_z:
                                gravity_z = top_z
                    final_z = float('inf')
                    valid_z_found = False
                    for zne in zones:
                        if (cx >= zne['x1'] - 0.01 and max_x <= zne['x2'] + 0.01 and
                                cy >= zne['y1'] - 0.01 and max_y <= zne['y2'] + 0.01):
                            pz_cand = max(gravity_z, zne.get('z1', 0))
                            if pz_cand + dz <= zne.get('z2', wh_hgt) + 0.001 and pz_cand < final_z:
                                final_z = pz_cand
                                valid_z_found = True
                    if valid_z_found:
                        center_x, center_y = cx + dx / 2, cy + dy / 2
                        dist = (center_x - target_x) ** 2 + (center_y - target_y) ** 2
                        floor_bias = 0 if final_z <= 0.01 else 1
                        score = (floor_bias, final_z, cy - zone_oy_cpu, cx - zone_ox_cpu, dist)
                        if best is None or score < best[7]:
                            best = (center_x, center_y, final_z, rot, dx, dy, dz, score)
                        if is_fast and final_z <= 0.01 and dist < 0.1:
                            return best
            return best

        # Dispatch to GPU/vectorized search when torch is available, CPU loop otherwise
        _pc = _placed_count
        if _TORCH_AVAILABLE:
            def _search(cands, is_fast, az=assigned_zone):
                return _search_candidates_gpu(
                    cands, rots, l, w, h,
                    _placed_buf, _pc, use_zones, az,
                    wh_len, wh_wid, wh_hgt, sort_tx + l / 2, sort_ty + w / 2,
                    is_fast, _dev)
        else:
            def _search(cands, is_fast, az=assigned_zone):
                return perform_search(cands, is_fast,
                                      check_zones=[az] if az is not None else None)

        best_pos = _search(search_candidates, fast_mode)

        # ── Multi-pass exhaustive retry: shrink grid step until a valid spot is found ──
        # Pass 1: 0.5 m lattice  (fast, already in search_candidates but re-run strict)
        # Pass 2: 0.25 m lattice (double density)
        # Pass 3: 0.10 m lattice (fine — catches items wedged by small neighbours)
        # Pass 4: 0.05 m lattice (near-pixel — absolute maximum density)
        # Each pass also includes adjacency touch-points from ALL placed items so
        # tight-fit positions next to every placed box are evaluated.
        if best_pos is None:
            _retry_zone = assigned_zone  # None → whole warehouse

            # Build the bounding rectangle to search within
            if _retry_zone is not None:
                _rx1, _ry1 = float(_retry_zone['x1']), float(_retry_zone['y1'])
                _rx2, _ry2 = float(_retry_zone['x2']), float(_retry_zone['y2'])
            else:
                _rx1, _ry1 = 0.0, 0.0
                _rx2, _ry2 = wh_len, wh_wid

            _GRID_STEPS = [0.5, 0.25, 0.10, 0.05]

            for _step in _GRID_STEPS:
                if best_pos is not None:
                    break

                _retry_cands = set()

                # 1. Full lattice over the target rectangle at this resolution
                _cx = _rx1
                while _cx <= _rx2 + _step * 0.5:
                    _cy = _ry1
                    while _cy <= _ry2 + _step * 0.5:
                        _retry_cands.add((_cx, _cy))
                        _cy = round(_cy + _step, 6)
                    _cx = round(_cx + _step, 6)

                # 2. Adjacency touch-points from EVERY placed item (not just last 100)
                for (px, py, pz, pdx, pdy, pdz) in placed_items:
                    _retry_cands.add((px + pdx, py))          # right edge
                    _retry_cands.add((px, py + pdy))          # back edge
                    _retry_cands.add((px + pdx, py + pdy))    # back-right corner
                    _retry_cands.add((px, py))                 # front-left corner
                    _retry_cands.add((px + pdx - l, py))      # flush-left align
                    _retry_cands.add((px, py + pdy - w))      # flush-front align

                # 3. Filter to valid non-zero-area positions inside target rect
                _valid_retry = [
                    (cx, cy) for (cx, cy) in _retry_cands
                    if cx >= _rx1 - 0.001 and cy >= _ry1 - 0.001
                    and cx + l <= _rx2 + 0.001 and cy + w <= _ry2 + 0.001
                    and cx >= 0 and cy >= 0
                ]

                if not _valid_retry:
                    continue

                # Sort front-left first (zone origin) for greedy bottom-left fill
                _retry_sorted = sorted(_valid_retry, key=lambda p: (
                    (p[0] - _rx1) ** 2 + (p[1] - _ry1) ** 2
                ))

                best_pos = _search(_retry_sorted, False, az=_retry_zone)

        # ── Last resort: try every OTHER zone if assigned zone is totally full ──
        if best_pos is None and assigned_zone is not None:
            for _fallback_zne in use_zones:
                if _fallback_zne is assigned_zone:
                    continue
                _fz_cands = set()
                _fx1, _fy1 = float(_fallback_zne['x1']), float(_fallback_zne['y1'])
                _fx2, _fy2 = float(_fallback_zne['x2']), float(_fallback_zne['y2'])
                _cx = _fx1
                while _cx <= _fx2:
                    _cy = _fy1
                    while _cy <= _fy2:
                        _fz_cands.add((_cx, _cy))
                        _cy = round(_cy + 0.25, 6)
                    _cx = round(_cx + 0.25, 6)
                for (px, py, pz, pdx, pdy, pdz) in placed_items:
                    _fz_cands.update([
                        (px + pdx, py), (px, py + pdy),
                        (px + pdx, py + pdy), (px, py),
                    ])
                _fz_valid = [
                    (cx, cy) for (cx, cy) in _fz_cands
                    if cx >= _fx1 - 0.001 and cy >= _fy1 - 0.001
                    and cx + l <= _fx2 + 0.001 and cy + w <= _fy2 + 0.001
                    and cx >= 0 and cy >= 0
                ]
                if _fz_valid:
                    _fz_sorted = sorted(_fz_valid, key=lambda p: (p[0] - _fx1) ** 2 + (p[1] - _fy1) ** 2)
                    best_pos = _search(_fz_sorted, False, az=_fallback_zne)
                if best_pos is not None:
                    break

        if best_pos is None and assigned_zone is None and fast_mode:
            best_pos = _search(sorted_candidates, False)


    
        # Apply placement
        if best_pos:
            b_x, b_y, b_z, b_rot, b_dx, b_dy, b_dz, _ = best_pos
        else:
            b_rot = solution[idx, 3] if not can_rotate else 0
            dims = get_rotated_dims(l, w, h, b_rot)
            b_dx, b_dy, b_dz = dims

            if assigned_zone is not None:
                # Zone-aware fallback: clamp within the assigned zone's bounds
                az_x1 = float(assigned_zone['x1'])
                az_y1 = float(assigned_zone['y1'])
                az_x2 = float(assigned_zone['x2'])
                az_y2 = float(assigned_zone['y2'])
                az_z1 = float(assigned_zone.get('z1', 0))
                az_z2 = float(assigned_zone.get('z2', wh_hgt))
                # Stack only on top of items already inside this zone
                zone_tops = [p[2] + p[5] for p in placed_items
                             if p[2] < 1000 and
                                p[0] + p[3] > az_x1 + 0.01 and p[0] < az_x2 - 0.01 and
                                p[1] + p[4] > az_y1 + 0.01 and p[1] < az_y2 - 0.01]
                b_z = max(zone_tops) if zone_tops else az_z1
                # Clamp so item never pierces the zone ceiling
                b_z = max(az_z1, min(b_z, az_z2 - b_dz))
                b_x = max(az_x1 + b_dx / 2, min(az_x2 - b_dx / 2, solution[idx, 0]))
                b_y = max(az_y1 + b_dy / 2, min(az_y2 - b_dy / 2, solution[idx, 1]))
            else:
                # Original zone-agnostic fallback
                b_z = 2000.0
                valid_placed = [p for p in placed_items if p[2] < 1000]
                if not valid_placed:
                    b_z = 0.0
                else:
                    max_top = max([p[2] + p[5] for p in valid_placed])
                    b_z = max_top
                b_x = max(b_dx / 2, min(wh_len - b_dx / 2, solution[idx, 0]))
                b_y = max(b_dy / 2, min(wh_wid - b_dy / 2, solution[idx, 1]))
            
        solution[idx, 0] = b_x
        solution[idx, 1] = b_y
        solution[idx, 2] = b_z
        solution[idx, 3] = b_rot
        
        placed_items.append((b_x - b_dx/2, b_y - b_dy/2, b_z, b_dx, b_dy, b_dz))
        grid.insert(len(placed_items)-1, b_x - b_dx/2, b_y - b_dy/2, b_x + b_dx/2, b_y + b_dy/2)
        if _placed_buf is not None:
            _placed_buf[_placed_count, 0] = float(b_x - b_dx/2)
            _placed_buf[_placed_count, 1] = float(b_y - b_dy/2)
            _placed_buf[_placed_count, 2] = float(b_z)
            _placed_buf[_placed_count, 3] = float(b_dx)
            _placed_buf[_placed_count, 4] = float(b_dy)
            _placed_buf[_placed_count, 5] = float(b_dz)
            _placed_count += 1

        # Periodic Callback for real-time visualization
        if callback and (len(placed_items) % callback_interval == 0 or len(placed_items) == num_items):
            callback(solution)

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

    items_props = np.zeros((num_items, 9))
    for i, item in enumerate(items):
        items_props[i] = [
            item['length'], item['width'], item['height'],
            item['can_rotate'], item['stackable'],
            item['access_freq'], item.get('weight', 0),
            hash(item.get('category', '')) % 10000,
            item.get('fragility', 0),
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
