# Repair Heuristic Algorithm

The Repair Heuristic is a greedy optimization algorithm that takes an initial item placement and "repairs" it to ensure all items are validly placed, following gravity and structural constraints.

## Placement Priority
- **Robust Items First**: Non-fragile, heavy items are placed first to provide a stable base.
- **Fragile Items Last**: Fragile items are placed on top of more robust ones.
- **Volume/Weight Descending**: Large/Heavy items are given priority for lower-level placement.

## Physics & Gravity
- **Gravity Calculation**: Calculates the lowest valid Z-coordinate for each item based on its horizontal overlap with items already placed.
- **Support Validation**: Ensures at least 20% of the item's area is supported by objects underneath it.

## Code Snippet (Python Implementation)

```python
def repair_solution_compact(solution, items_props, warehouse_dims):
    """
    Repair solution by placing items in valid positions with gravity.
    
    1. Sort items by fragility, weight, and volume.
    2. Iteratively place each item at its 'physics-correct' Z-position.
    3. Use a Spatial Grid index to quickly query for supporting items.
    """
    placed_items = []
    grid = SimpleGrid(wh_len, wh_wid)

    for idx in sorted_indices:
        # Find gravity Z for item at target (x, y)
        gravity_z = 0.0
        potential_supporters = grid.query(min_x, min_y, max_x, max_y)
        
        for p_idx in potential_supporters:
             # Calculate intersection height (top_z = pz + pdz)
             # ...
             if overlap:
                 gravity_z = max(gravity_z, top_z)
        
        # Apply placement with gravity
        solution[idx, 2] = gravity_z
        placed_items.append(...)
        grid.insert(...)
        
    return solution
```
