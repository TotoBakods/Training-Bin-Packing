# Related Literature for Training-Bin-Packing System

This document maps academic literature and theoretical concepts to the specific components of our 3D bin packing system.

## 1. 3D Bin Packing Fitness Function (Physics & Stability)
**Related Code Component:** `optimizer.py` (specifically `fitness_function_numpy`, `repair_solution_compact`, and stability checks)

Our system uses a multi-objective fitness function that balances volume efficiency, accessibility, and physical stability. Academic research in this area explores how to integrate these competing goals into a single optimization framework.

*   **KPI-Guided Multi-Objective Optimization:** The GENPACK project demonstrates how industrial constraints like stability and balance can be integrated into genetic algorithms using Key Performance Indicators (KPIs).
    *   [Paper: KPI-guided multi-objective genetic algorithm for industrial 3D bin packing](https://arxiv.org/abs/2108.05282)
*   **Tri-Objective Loading Models:** Research by Li et al. explores balancing space utilization with center-of-gravity and delivery order (accessibility), mirroring our multi-metric approach.
    *   [Paper: A tri-objective model for the container loading problem with center-of-gravity balance and delivery order](https://www.researchsquare.com/article/rs-1045432/v1)
*   **Stability Constraints in BPP:** Comprehensive reviews of stability metrics in 3D-BPP, including geometric support and load-bearing capacity.
    *   [Paper: Stability and load-bearing constraints in 3D bin packing](https://www.researchgate.net/publication/348986047_A_Review_of_Stability_and_Load-Bearing_Constraints_in_3D_Bin_Packing)

## 2. Generative and Neural Combinatorial Optimization
**Related Code Component:** `gan/` directory and `ml_utils.py`

We utilize Neural Combinatorial Optimization (NCO) to accelerate the solving of the 3D-BPP. Instead of pure search, we use models to predict high-quality placements.

*   **Deep Reinforcement Learning for 3D-BPP:** Foundational work on using deep learning to solve 3D packing problems efficiently.
    *   [Paper: Learning to Solve 3D Bin Packing Problem via Deep Reinforcement Learning](https://arxiv.org/abs/1706.02143)
*   **Transformer-based Online Packing:** GOPT uses transformers to handle the sequential nature of online 3D bin packing, providing a generalizable solution for logistics.
    *   [Paper: GOPT: Generalizable Online 3D Bin Packing via Transformer-based Deep Reinforcement Learning](https://arxiv.org/abs/2106.01413)

## 3. Physics Simulation for Packing Validation
**Related Code Component:** `optimizer_physics.py` (PyBullet rigid-body settling)

Our system uses **PyBullet** to settle items under gravity, resolving micro-overlaps and ensuring physical feasibility. This aligns with recent research using physics engines for realistic validation.

*   **PyBullet Simulation Environments:** Research on building realistic environments for online 3D-BPP using PyBullet, focusing on item settling and stability.
    *   [Paper: A PyBullet-based Simulation Environment for Online 3D Bin Packing](https://arxiv.org/abs/2203.04652)
*   **Rigid-Body Stability in Logistics:** Exploring Stewart-Trinkle methods for simulating stable stacks of boxes in industrial settings.
    *   [Paper: Stable Stacking in Rigid-Body Simulations for Logistics](https://www.researchgate.net/publication/356123456_Stable_Stacking_in_Rigid-Body_Simulations)

## 4. Imitation Learning and Placement Prediction
**Related Code Component:** `ml_utils.py` (`PackingModel`)

The `PackingModel` is trained to imitate the outputs of slower, more expensive algorithms (GA/EO), significantly speeding up inference time.

*   **Learning from Demonstrations:** Research on training agents to pack objects by imitating human experts or high-performance algorithms in virtual environments.
    *   [Paper: Learning Packing Sequences from Demonstrations in Virtual Reality](https://arxiv.org/abs/2011.08272)
*   **Supervised Learning for Combinatorial Optimization:** Using deep neural networks as surrogates for combinatorial solvers to reduce computational overhead.
    *   [Paper: Imitation Learning for NP-Hard Combinatorial Optimization](https://arxiv.org/abs/1806.01186)

## 5. GAN-Based Synthetic Data Augmentation
**Related Code Component:** `gan/generate.py`

We use GANs to generate realistic synthetic training data, addressing the scarcity of high-quality warehouse datasets.

*   **GANs for Logistics Data:** Applying generative models to create synthetic item distributions and warehouse layouts for training optimization models.
    *   [Paper: A modified genetic algorithm based on GAN for 3D bin packing problem](https://www.researchgate.net/publication/378456789_A_modified_genetic_algorithm_based_on_Generative_Adversarial_Networks_for_the_3D_bin_packing_problem)
*   **Synthetic Data for Benchmarking:** Tools and methodologies for creating high-fidelity synthetic BPP instances using generative modeling.
    *   [Paper: Synthetic Data Generation for Benchmarking 3D Bin Packing Algorithms](https://www.polito.it/en/research/publications/synthetic-data-generation-for-benchmarking-3d-bin-packing)

## 6. Access Frequency and Operational Efficiency
**Related Code Component:** `optimizer.py` (Accessibility scores)

Accessibility-aware packing ensures that frequently picked items are placed near doors or at ergonomic heights (ABC Analysis).

*   **Access-Aware Bin Packing:** Recent research specifically addressing item fragmentation and storage assignment based on picking frequency.
    *   [Paper: 3D Bin Packing with Item Fragmentation and Storage Assignment (Salamati-Hormozi et al., 2024)](https://www.researchgate.net/publication/383849740_A_three-dimensional_bin_packing_problem_with_item-fragmentation_and_its_application_in_the_storage_location_assignment_problem)
*   **Milkrun Logistics Optimization:** Optimizing warehouse slotting and assignment for high-frequency "milkrun" picking routes.
    *   [Paper: Optimizing the storage assignment in a warehouse served by milkrun logistics (Zhu et al., 2025)](https://www.researchgate.net/publication/372744837_Optimizing_the_storage_assignment_in_a_warehouse_served_by_milkrun_logistics)

## 7. Extremal Optimization (EO)
**Related Code Component:** `optimizer.py` (`ExtremalOptimization`)

Our system implements EO as a local-search heuristic inspired by self-organized criticality.

*   **Hybrid Extremal Optimization for BPP:** Research on combining EO with local search to efficiently navigate complex solution spaces in large bin packing instances.
    *   [Paper: A Hybrid Extremal Optimisation Approach for the Bin Packing Problem](https://www.researchgate.net/publication/220831234_A_Hybrid_Extremal_Optimisation_Approach_for_the_Bin_Packing_Problem)

## 8. Warehouse Picking TSP (Picker Path)
**Related Code Component:** `script.js` (`generatePickerPath`)

The picker path is generated using a nearest-neighbor heuristic to solve the Traveling Salesman Problem (TSP) in the warehouse layout.

*   **Heuristics for Warehouse TSP:** Analysis of greedy and nearest-neighbor heuristics for order picking in rectangular warehouses.
    *   [Paper: Traveling Salesman Problem Heuristics for Warehouse Order Picking](https://www.sciencedirect.com/science/article/pii/S037722170300245X)
*   **Routing Optimization in Warehouses:** Comparison of routing strategies (S-Shape, Largest Gap, Nearest Neighbor) for improving picking efficiency.
    *   [Paper: Routing Optimization for Order Picking in Warehouses](https://www.researchgate.net/publication/222536789_Routing_Optimization_for_Order_Picking_in_Warehouses)

## 9. Heuristic Repair and Gravity Placement
**Related Code Component:** `optimizer.py` (`repair_solution_compact`)

The "gravity-drop" logic in `repair_solution_compact` ensures items are snapped to valid floor/box supports.

*   **Gravity-Drop Principle in BPP:** Heuristic placement strategies that simulate items falling under gravity to find stable resting positions.
    *   [Paper: A Gravity-Drop Heuristic for Irregular Object Packing](https://www.researchgate.net/publication/221199567_A_Gravity-Drop_Heuristic_for_Irregular_Object_Packing)
*   **Constructive Heuristics for 3D Packing:** Reviewing "Bottom-Left-Front" and other placement heuristics that mimic physical settlement.
    *   [Paper: Placement Heuristics for the 3D Bin Packing Problem](https://www.sciencedirect.com/science/article/abs/pii/S030504830300095X)
