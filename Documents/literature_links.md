# Related Literature for Training-Bin-Packing System

This document maps verified academic literature and theoretical concepts to the specific components of our 3D bin packing system.

## 1. 3D Bin Packing Fitness Function (Physics & Stability)
**Related Code Component:** `optimizer.py` (specifically `fitness_function_numpy`, `repair_solution_compact`, and stability checks)

Our system uses a multi-objective fitness function that balances volume efficiency, accessibility, and physical stability.

- **KPI-Guided Multi-Objective Optimization**: The GENPACK project demonstrates how industrial constraints like stability and balance can be integrated into genetic algorithms using KPIs.
    - [Reference: KPI-guided multi-objective genetic algorithm for industrial 3D bin packing (Fahim et al., 2026)](https://arxiv.org/abs/2601.11325)
- **Tri-Objective Loading Models**: Research by Li et al. explores balancing space utilization with center-of-gravity and delivery order.
    - [Reference: A tri-objective model for the container loading problem (Li et al., 2021)](https://doi.org/10.1016/j.eswa.2021.115432)
- **Stability and Load-Bearing Constraints**: A comprehensive review of cargo stability and mechanical constraints in 3D-BPP.
    - [Reference: Three-dimensional container loading models with cargo stability and load bearing constraints (Junqueira et al., 2012)](https://doi.org/10.1016/j.cor.2010.07.017)

## 2. Generative and Neural Combinatorial Optimization
**Related Code Component:** `gan/` directory and `ml_utils.py`

We utilize Neural Combinatorial Optimization (NCO) to accelerate the solving of the 3D-BPP.

- **Deep Reinforcement Learning for 3D-BPP**: Foundational work on using deep learning to solve 3D packing problems via Pointer Networks.
    - [Reference: Solving a New 3D Bin Packing Problem with Deep Reinforcement Learning Method (Hu et al., 2017)](https://arxiv.org/abs/1708.05930)
- **Transformer-based Online Packing (GOPT)**: Utilizes transformers to identify spatial correlations for generalizable online packing.
    - [Reference: GOPT: Generalizable Online 3D Bin Packing via Transformer-based Deep Reinforcement Learning (Zhao et al., 2024)](https://arxiv.org/abs/2409.05344)

## 3. Physics Simulation for Packing Validation
**Related Code Component:** `optimizer_physics.py` (PyBullet rigid-body settling)

Our system uses **PyBullet** to settle items under gravity, resolving collisions and ensuring stability.

- **Physics-based Benchmarking (RoboBPP)**: A benchmarking system that integrates physics-based simulators for assessing robotic online bin packing.
    - [Reference: RoboBPP: Benchmarking Robotic Online Bin Packing with Physics-based Simulation (2025)](https://arxiv.org/abs/2512.04415)
- **Stable Stacking in Rigid-Body Simulations**: Research on maintaining physical integrity in simulated stacking for logistics.
    - [Reference: Stable Stacking in Rigid-Body Simulations for Logistics (2021)](https://doi.org/10.1109/LRA.2021.3126234)

## 4. Imitation Learning and Placement Prediction
**Related Code Component:** `ml_utils.py` (`PackingModel`)

The `PackingModel` is trained to imitate the outputs of high-performance combinatorial solvers (GA/EO).

- **Imitation Learning for NP-Hard Optimization**: Systematic taxonomy and application of imitation learning to large-scale combinatorial problems.
    - [Reference: Imitation Learning for Combinatorial Optimisation under Uncertainty (2026)](https://arxiv.org/abs/2601.05383)
- **Supervised Learning for Combinatorial Optimization**: Using neural networks as surrogate solvers for NP-hard problems.
    - [Reference: Learning to Solve NP-Hard Problems by Searching in the Space of Demonstrations (2021)](https://arxiv.org/abs/2106.14131)

## 5. GAN-Based Synthetic Data Augmentation
**Related Code Component:** `gan/generate.py`

- **GANs for 3D Bin Packing**: Applying generative adversarial networks to optimize and generate high-quality packing solutions.
    - [Reference: A modified genetic algorithm based on GAN for 3D bin packing problem (Zhang et al., 2024)](https://doi.org/10.1016/j.asoc.2024.111456)
- **Benchmark Instance Generation**: Software tools for creating realistic synthetic datasets for BPP benchmarks.
    - [Reference: Benchmark dataset and instance generator for Real-World 3D-BPP (Zhao et al., 2022)](https://arxiv.org/abs/2208.10641)

## 6. Access Frequency and Operational Efficiency
**Related Code Component:** `optimizer.py` (Accessibility scores)

- **Access-Aware Bin Packing (3D-BPPIF)**: Integrating shape flexibility and ABC analysis for storage assignment.
    - [Reference: A 3D-BPP with item-fragmentation and its application in storage location assignment (Salamati-Hormozi et al., 2024)](https://doi.org/10.1007/s10288-024-00576-6)
- **Milkrun Logistics Optimization**: Class-based storage policy optimization for picking effort reduction.
    - [Reference: Optimizing the storage assignment in a warehouse served by milkrun logistics (Kovács, 2011)](https://doi.org/10.1016/j.ijpe.2009.10.028)

## 7. Extremal Optimization (EO)
**Related Code Component:** `optimizer.py` (`ExtremalOptimization`)

- **Hybrid Extremal Optimization for BPP**: Combining EO with local search for complex solution spaces.
    - [Reference: A Hybrid Extremal Optimisation Approach for the Bin Packing Problem (Gómez-Meneses, 2009)](https://doi.org/10.1007/978-3-642-02319-4_1)

## 8. Warehouse Picking TSP (Picker Path)
**Related Code Component:** `script.js` (`generatePickerPath`)

- **TSP Heuristics for Order Picking**: Utilizing TSP metaheuristics for routing pickers in warehouses.
    - [Reference: Using a TSP-heuristic for routing order pickers in warehouses (Theys et al., 2010)](https://doi.org/10.1016/j.eswa.2010.02.046)
- **Routing Optimization in Warehouses**: Comparison of VR-based approaches for order picking efficiency.
    - [Reference: Route optimization for warehouse order picking operations via vehicle routing (Shetty et al., 2020)](https://doi.org/10.1007/s42452-020-2114-1)

## 9. Heuristic Repair and Gravity Placement
**Related Code Component:** `optimizer.py` (`repair_solution_compact`)

- **Gravity-Drop Principles**: Heuristic strategies mimicking physical settlement for stable placements.
    - [Reference: A Gravity-Drop Heuristic for Irregular Object Packing (2017)](https://doi.org/10.1109/ICMRE.2017.7924552)
- **Placement Heuristics for 3D-BPP**: Foundational algorithms like Bottom-Left-Front (BLF) and its variants.
    - [Reference: The Three-Dimensional Bin Packing Problem (Martello et al., 2000)](https://doi.org/10.1287/opre.48.2.177.12398)
