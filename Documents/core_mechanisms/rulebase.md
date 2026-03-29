# Core Logistics Rulebase

The "rulebase" for the warehouse optimizer governs how items are sorted, where they can be placed, and how they interact with other items such as gravity, stacking, and fragility.

## 1. Item Sorting & Placement Priority
Before placement, items are sorted to ensure heavy and robust items form the base, while fragile items are placed last (on top).

```python
# From optimizer.py -> repair_solution_compact
# Sort: fragile last (0=robust, 1=fragile), then heavy and large first
fragility = items_props[:, 8]
weights = items_props[:, 6]
volumes = items_props[:, 0] * items_props[:, 1] * items_props[:, 2]

indices = np.arange(num_items)
sorted_indices = sorted(indices, key=lambda i: (fragility[i], -weights[i], -volumes[i], i))
```

## 2. Stacking & Support Rules
These rules prevent items from floating and enforce structural integrity by requiring a minimum support area.

```python
# From optimizer.py -> calculate_z_for_item
# 1. Reject stacking on non-stackable items
if strict_stacking and other_items_stackable[overlaps] == 0:
    return 1000000.0 # Effectively impossible Z

# 2. Stability check: ensure at least 20% support area
supported_area = np.sum(intersection_area)
if supported_area < (item_area * 0.2): 
    return max_z + 100000.0 # Force placement elsewhere
```

## 3. AABB Collision Detection
The system uses Axis-Aligned Bounding Box (AABB) checks to prevent item overlaps in 3D space.

```python
# Vectorized Overlap Check
overlap_x = np.abs(x1 - x2) < (hw1 + hw2 - 0.01) # 1cm tolerance
overlap_y = np.abs(y1 - y2) < (hh1 + hh2 - 0.01)
overlap_z = (z_top1 > z_bottom2 + 0.01) & (z_bottom1 < z_top2 - 0.01)

is_colliding = overlap_x & overlap_y & overlap_z
```

## 4. Exclusion Zone Rules
Prevents items from being placed in restricted areas (e.g., walkways, fire exits).

```python
# Center-to-center distance must be greater than combined radii
dx = np.abs(item_x - zone_center_x)
dy = np.abs(item_y - zone_center_y)

if dx < (item_radius + zone_half_width) and dy < (item_radius + zone_half_height):
    apply_penalty() # Item is inside an exclusion zone
```
