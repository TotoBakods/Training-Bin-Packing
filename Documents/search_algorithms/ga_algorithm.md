# Genetic Algorithm (GA) Optimization

The Genetic Algorithm (GA) is a population-based heuristic inspired by the process of natural selection. In this project, the "GA" variant refers to a Machine Learning model (Neural Network) trained via imitation learning to mimic the behavior of a Genetic Algorithm search.

## Overview
- **Type**: Imitation Learning (ML Model)
- **Training Target**: Historically successful GA-generated packing layouts.
- **Strength**: Good at discovering global structures and maintaining diversity in item placements.

## Implementation Details
The GA variant uses a pre-trained PyTorch model (`fit_ga.pth`) to predict (x, y, z, rotation) for each item. The raw predictions are then refined by a physics-based repair heuristic to ensure stability.

## Code Snippet (Backend Integration)

```python
@app.route('/api/optimize/ga', methods=['POST'])
def optimize_ga():
    """Triggers the GA-imitation Neural Network optimizer."""
    # ... setup and weight extraction ...
    
    # Load the GA-trained ML Optimizer
    optimizer = MLOptimizer("fit_ga")
    
    # Run Inference
    best_solution, best_fitness, time_to_best = optimizer.optimize(
        items, warehouse, weights, 
        callback=update_progress, 
        optimization_state=optimization_state
    )
    
    # Finalize and repair via Physics Settlement
    finalize_optimization(best_solution, 'ML - Genetic Algorithm', ...)
```
