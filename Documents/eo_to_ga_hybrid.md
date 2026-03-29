# Hybrid EO-GA Optimization

The Hybrid EO-GA approach focuses on the initial identification of "extreme" items (poor fits) and their resolution before broadly refining the layout using a population-based genetic strategy.

## Overview
- **Type**: Hybrid Imitation learning.
- **Strategy**: Local search optimization followed by global exploration for refinement.
- **Goal**: Resolve immediate spatial conflicts before iterating toward a higher-quality global optimum.

## Implementation Details
The machine learning model (`fit_eo_ga.pth`) is trained on datasets where this specific order of operations (EO before GA) was followed to generate the target layout.

## Code Snippet (Backend Integration)

```python
@app.route('/api/optimize/eo-ga', methods=['POST'])
def optimize_hybrid_eo_ga():
    """Hybrid optimizer: Triggers the EO-then-GA imitation Neural Network."""
    # ... setup and weight extraction ...
    
    # Load the Hybrid EO+GA-trained ML Optimizer
    optimizer = MLOptimizer("fit_eo_ga")
    
    # Run Inference
    best_solution, best_fitness, time_to_best = optimizer.optimize(
        items, warehouse, weights, 
        callback=update_progress, 
        optimization_state=optimization_state
    )
    
    # Finalize and repair via Physics Settlement
    finalize_optimization(best_solution, 'ML - Hybrid EO-GA', ...)
```
