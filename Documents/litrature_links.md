# Related Literature Review (RRL): Model Metrics

This document provides theoretical justification and research context for the metrics used to evaluate the 3D Bin Packing models in this project, mapping them to state-of-the-art (SOTA) research.

## 1. Spatial Coordinate Regression (R² Metrics)
Evaluates the model's accuracy in predicting precise (x, y, z) coordinates.
- **Metric**: R² x, y, z, rot.
- **Project Results**: High success in Z-axis (R² ≈ 0.90), but challenges in horizontal (x, y) coordinates.
- **RRL Context**: 
    - Standard regression metrics (MAE/MSE/R²) are used to evaluate "Spatial Intelligence" in neural-heuristic hybrids.
- **Reference**: 
    - [Solving a New 3D Bin Packing Problem with Deep Reinforcement Learning Method](https://arxiv.org/abs/1708.05930) (Haoyuan Hu et al., 2017) - Discusses coordinate-based optimization using Pointer Networks.

## 2. Vertical Stacking & Stability
Justifies the "Z-Floor %" and "Stability" metrics.
- **Metrics**: Z Floor %, Z Low %, Z High %, Stability Score.
- **Project Results**: ~94.5% Z-Floor success; ~1.00 Stability.
- **RRL Context**: 
    - **GENPACK (2021)** emphasizes "Surface Support".
    - **Physical Stability** is a core constraint in real-world benchmarks like BED-BPP.
- **Reference**: 
    - [Online 3D Bin Packing with Constrained Deep Reinforcement Learning](https://arxiv.org/abs/2006.14978) (Zhao et al., 2020) - Focuses on stability and constraint-aware placement.

## 3. Category Clustering (m)
Justifies the grouping of items by category.
- **Metric**: Cat Cluster (m).
- **Project Results**: Avg clustering distance 8.9m - 9.4m.
- **RRL Context**:
    - **The Price of Clustering (PoC)**: Theoretical bounds on the number of bins required when items are segregated by cluster.
- **Reference**: 
    - [The Price of Clustering in Bin Packing](https://arxiv.org/abs/1908.06727) (Azar et al., 2019) - Provides theoretical bounds for clustered packing.

## 4. Accessibility (ABC Analysis)
Justifies the "Access" metric in the fitness function.
- **Metric**: Access score (0.0 - 1.0).
- **Project Results**: ~0.08 average access (prioritizing distance to door).
- **RRL Context**:
    - Logistics research integrates warehouse travel distance and "Easy-to-Reach" zones into the packing objective.
- **Reference**: 
    - [A Survey on Bin Packing Problems](https://arxiv.org/abs/2203.04787) - Discusses various industrial constraints and objective functions.

## 5. Fragility & Stacking Constraints
Justifies "Fragile OK %".
- **Metric**: Fragile OK % (Compliance).
- **Project Results**: 100% compliance.
- **RRL Context**:
    - Real-world industrial constraints require "Load-Bearing" checks where item fragility prevents certain stacking sequences.
- **Reference**: 
    - [Online 3D Bin Packing with Constrained Deep Reinforcement Learning](https://arxiv.org/abs/2006.14978) (Zhao et al., 2020).

## 6. Space Utilization & Fitness
Standard BPP objectives.
- **Metrics**: Space %, Fitness.
- **Project Results**: ~4.26% Space Utilization (Warehouse-scale).
- **RRL Context**: 
    - Literature utilization typically focuses on *full bin density*.
- **Reference**: 
    - [A Literature Review on the Bin Packing Problem](https://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1405-55462021000400002).

## 7. Inference vs. Repair Efficiency
- **Metric**: Inference ms vs. Total ms.
- **Project Results**: ~1.5ms Inference; ~57s Repair.
- **RRL Context**: 
    - Many DRL models use a "Predict-then-Fix" or "Predict-then-Project" approach to ensure feasibility.
- **Reference**: 
    - [Solving a New 3D Bin Packing Problem with Deep Reinforcement Learning Method](https://arxiv.org/abs/1708.05930).
