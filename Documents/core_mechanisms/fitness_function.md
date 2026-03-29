# Fitness Function Algorithm

The Fitness Function is used to evaluate the quality of a given bin-packing solution. It considers multiple factors including space utilization, item accessibility, and structural stability.

## Key Metrics
- **Space Utilization**: Ratio of total item volume to warehouse volume.
- **Accessibility**: weighted average of item proximity to the door based on access frequency.
- **Stability**: Measures if items are properly supported by the floor or other items below them.
- **Exclusion Zones**: Penalties applied for items placed within restricted areas.

## Code Snippet (NumPy Implementation)

```python
def fitness_function_numpy(solution, items_props=None, warehouse_dims=None, weights=None, valid_z=None, exclusion_zones_arr=None):
    """
    Calculates the fitness of a warehouse layout.
    
    Args:
        solution: (N, 4) array of [x, y, z, rotation]
        items_props: (N, 9) array of item properties
        warehouse_dims: (L, W, H, DoorX, DoorY)
        weights: Dictionary of weight factors for each metric
    """
    # Calculate Space Utilization
    total_vol = np.sum(items_props[:, 0] * items_props[:, 1] * items_props[:, 2])
    wh_vol = warehouse_dims[0] * warehouse_dims[1] * warehouse_dims[2]
    space_util = total_vol / wh_vol if wh_vol > 0 else 0
    
    # Calculate Accessibility (Distance to door)
    door_x, door_y = warehouse_dims[3], warehouse_dims[4]
    dists = np.sqrt((solution[:, 0] - door_x)**2 + (solution[:, 1] - door_y)**2)
    access_scores = 1.0 / (1.0 + dists)
    
    freqs = items_props[:, 5]
    accessibility = np.average(access_scores, weights=freqs) if np.sum(freqs) > 0 else np.mean(access_scores)
    
    # Stability Check (Simplified support logic)
    # ... (Vectorized support checks omitted for brevity) ...
    
    # Final weight calculation
    fitness = (space_util * weights['space'] + 
               accessibility * weights['accessibility'] + 
               stability * weights['stability'])
               
    return fitness, space_util, accessibility, stability
```
