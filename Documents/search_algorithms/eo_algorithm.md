# Extremal Optimization (EO) Variant

Extremal Optimization (EO) is a local search heuristic that improves a solution by iteratively removing the "worst" component (e.g., items with the lowest individual fitness).

## Overview
- **Type**: Imitation Learning (ML Model)
- **Training Target**: EO-generated local-search data.
- **Strength**: High precision in localized placement and filling small gaps.

## Implementation Details
The EO variant utilizes a specialized Neural Network model trained to recognize and mimic the local-search behavior of Extremal Optimization.

## Code Snippet (Backend Integration)

```python
@app.route('/api/optimize/eo', methods=['POST'])
def optimize_eo():
    """Triggers the EO-imitation Neural Network optimizer."""
    # ... setup and weight extraction ...
    
    # Load the EO-trained ML Optimizer
    optimizer = MLOptimizer("fit_eo")
    
    # Run Inference
    best_solution, best_fitness, time_to_best = optimizer.optimize(
        items, warehouse, weights, 
        callback=update_progress, 
        optimization_state=optimization_state
    )
    
    # Finalize and repair via Physics Settlement
    finalize_optimization(best_solution, 'ML - Extremal Opt', ...)
```
