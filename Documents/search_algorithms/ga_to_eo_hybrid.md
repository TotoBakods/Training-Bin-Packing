# Hybrid GA-EO Optimization

The Hybrid GA-EO algorithm combines the global exploration of a Genetic Algorithm (GA) followed by the local refinement of Extremal Optimization (EO).

## Overview
- **Type**: Hybrid Imitation learning.
- **Strategy**: GA for broad initial search + EO for microscopic refinement.
- **Goal**: Find high-quality global layouts and then settle them with local precision.

## Implementation Details
The model (`fit_ga_eo.pth`) is trained on solutions that have undergone both search phases, aiming to achieve higher overall fitness than either method individually.

## Code Snippet (Backend Integration)

```python
@app.route('/api/optimize/ga-eo', methods=['POST'])
def optimize_hybrid():
    """Triggers the Hybrid GA-EO-imitation Neural Network optimizer."""
    # ... setup and weight extraction ...
    
    # Load the Hybrid GA+EO-trained ML Optimizer
    optimizer = MLOptimizer("fit_ga_eo")
    
    # Run Inference
    best_solution, best_fitness, time_to_best = optimizer.optimize(
        items, warehouse, weights, 
        callback=update_progress, 
        optimization_state=optimization_state
    )
    
    # Finalize and repair via Physics Settlement
    finalize_optimization(best_solution, 'ML - Hybrid GA-EO', ...)
```
