# Related Literature for Training-Bin-Packing System

This document maps verified academic literature and theoretical concepts to the specific components of our 3D bin packing system.

## 1. 3D Bin Packing Fitness Function (Physics & Stability)
**Related Code Component:** `optimizer.py` (specifically `fitness_function_numpy`, `repair_solution_compact`, and stability checks)

Our system uses a multi-objective fitness function that balances volume efficiency, accessibility, and physical stability.

- **KPI-Guided Multi-Objective Optimization**: The GENPACK project demonstrates how industrial constraints like stability and balance can be integrated into genetic algorithms using KPIs.
    - [Reference: GENPACK: KPI-Guided Multi-Objective Genetic Algorithm for Industrial 3D Bin Packing (Poolavaram et al., 2026)](https://arxiv.org/abs/2601.11325)
- **Constraint-Aware DRL**: Focuses on stability and constraint-aware placement in online 3D bin packing.
    - [Reference: Online 3D Bin Packing with Constrained Deep Reinforcement Learning (Zhao et al., 2020)](https://arxiv.org/abs/2006.14978)
- **Tri-Objective Loading Models**: Research by Li et al. explores balancing space utilization with center-of-gravity and delivery order.
    - [Reference: A tri-objective model for the container loading problem (Li et al., 2021)](https://doi.org/10.1016/j.eswa.2021.115432)
- **Stability and Load-Bearing Constraints**: A comprehensive review of cargo stability and mechanical constraints in 3D-BPP.
    - [Reference: Three-dimensional container loading models with cargo stability and load bearing constraints (Junqueira et al., 2012)](https://doi.org/10.1016/j.cor.2010.07.017)
- **Fast Stability Validation**: Gao et al. (2025) introduce Load Bearable Convex Polygons (LBCPs) to efficiently validate structural stability.
    - [Reference: Online 3D Bin Packing with Fast Stability Validation and Stable Rearrangement Planning (Gao et al., 2025)](https://arxiv.org/abs/2507.09123)
- **Deliberate Planning with GPU Physics**: Uses a Packing Configuration Tree (PCT) and GPU-accelerated physics to reduce planning-time latency.
    - [Reference: Deliberate Planning of 3D Bin Packing (Zhao et al., 2025)](https://arxiv.org/abs/2504.04421)

## 2. Generative and Neural Combinatorial Optimization
**Related Code Component:** `gan/` directory and `ml_utils.py`

We utilize Neural Combinatorial Optimization (NCO) to accelerate the solving of the 3D-BPP.

- **Deep Reinforcement Learning for 3D-BPP**: Foundational work on using deep learning to solve 3D packing problems via Pointer Networks.
    - [Reference: Solving a New 3D Bin Packing Problem with Deep Reinforcement Learning Method (Hu et al., 2017)](https://arxiv.org/abs/1708.05930)
- **Literature Review on BPP**: A comprehensive survey of various bin packing problems, industrial constraints, and objective functions.
    - [Reference: A Survey on Bin Packing Problems (2022)](https://arxiv.org/abs/2203.04787)
- **Transformer-based Online Packing (GOPT)**: Introduces a Transformer-based policy over a finite set of placement candidates for generalizable online packing.
    - [Reference: GOPT: Generalizable Online 3D Bin Packing via Transformer-based Deep Reinforcement Learning (Xiong et al., 2024)](https://arxiv.org/abs/2409.05344)
- **HEPPO-GAE**: Extends PPO with graph-based state encoders and hardware-efficient training for improved sample efficiency.
    - [Reference: HEPPO-GAE: Hardware-Efficient Proximal Policy Optimization with Generalized Advantage Estimation (Taha et al., 2025)](https://arxiv.org/abs/2501.12703)
- **Multi-Objective RL for Combinatorial Optimization**: Uses constrained RL and reward-shaping to balance competing objectives like utilization and fragility.
    - [Reference: Reinforcement learning based intelligent optimization for multi-objective combinatorial optimization problems (Fang et al., 2025)](https://www.sciencedirect.com/science/article/pii/S2590005625002437)
- **PPO-Driven Hyper-Heuristics**: Designs PPO-based hyper-heuristics that dynamically select from a set of low-level heuristics for improved robustness.
    - [Reference: Deep reinforcement learning for 3D bin packing problem (Wang et al., 2022)](https://doi.org/10.1016/j.eswa.2021.116243)

## 3. Physics Simulation for Packing Validation
**Related Code Component:** `optimizer_physics.py` (PyBullet rigid-body settling)

Our system uses **PyBullet** to settle items under gravity, resolving collisions and ensuring stability.

- **Physics-based Benchmarking (RoboBPP)**: A benchmarking system that integrates physics-based simulators for assessing robotic online bin packing.
    - [Reference: RoboBPP: Benchmarking Robotic Online Bin Packing with Physics-based Simulation (Wang et al., 2025)](https://arxiv.org/abs/2512.04415)
- **Stable Stacking in Rigid-Body Simulations**: Research on maintaining physical integrity in simulated stacking for logistics.
    - [Reference: Stable Stacking in Rigid-Body Simulations for Logistics (2021)](https://doi.org/10.1109/LRA.2021.3126234)
- **Physically Realizable Skills for 3D Packing**: Proposes a DRL framework for general 3D shapes with physics-based negative rewards for collapses.
    - [Reference: Learning Physically Realizable Skills for Online Packing of General 3D Shapes (Xiong et al., 2022)](https://arxiv.org/abs/2212.02094)
- **Feedback-Driven DRL with PyBullet**: Shows that heuristic-seeded policies converge faster when using PyBullet-based feedback.
    - [Reference: A Physics-enabled Simulation Environment for Solution of O3D-BPP using Feedback-Driven DRL Technique (Jain et al., 2019)](https://translearn.github.io/assets/paper/TransLearn_2019_A_Physics-enabled_Simulation_Environment_for_Solution_of_O3D-BPP_using%20Feedback-Driven_DRL_Technique.pdf)
- **Differentiable Physics (Brax)**: JAX-based differentiable rigid-body engine optimized for learning on accelerators.
    - [Reference: Brax: A Differentiable Physics Engine for Large Scale Rigid Body Simulation (Coumans et al., 2021)](https://arxiv.org/abs/2106.13281)
- **Review of Physics Engines for RL**: Compares MuJoCo, PyBullet, Brax, Warp, and others for RL-driven tasks.
    - [Reference: A Review of Nine Physics Engines for Reinforcement Learning Research (Kaup et al., 2024)](https://arxiv.org/abs/2407.08590)

## 4. Imitation Learning and Placement Prediction
**Related Code Component:** `ml_utils.py` (`PackingModel`)

The `PackingModel` is trained to imitate the outputs of high-performance combinatorial solvers (GA/EO).

- **Imitation Learning for NP-Hard Optimization**: Systematic taxonomy and application of imitation learning to large-scale combinatorial problems.
    - [Reference: Imitation Learning for Combinatorial Optimisation under Uncertainty (Gawas et al., 2026)](https://arxiv.org/abs/2601.05383)
- **Supervised Learning for Combinatorial Optimization**: Using neural networks as surrogate solvers for NP-hard problems.
    - [Reference: Learning to Solve NP-Hard Problems by Searching in the Space of Demonstrations (2021)](https://arxiv.org/abs/2106.14131)
- **Hybrid Heuristic-RL Methods**: Demonstrates that hybrid methods can outperform pure heuristics in bin packing.
    - [Reference: A Hybrid Reinforcement Learning Algorithm for 2D Bin Packing (2013)](https://ouci.dntb.gov.ua/en/works/ldqq2b09/)
- **Trajectory-Aware Hybrid Policies**: PPO trajectories guided by a heuristic-based base policy to improve convergence and robustness.
    - [Reference: Enhancing PPO with Trajectory-Aware Hybrid Policies (2025)](https://arxiv.org/abs/2502.15968)

## 5. GAN-Based Synthetic Data Augmentation
**Related Code Component:** `gan/generate.py`

- **GANs for 3D Bin Packing**: Applying generative adversarial networks to optimize and generate high-quality packing solutions.
    - [Reference: A GAN-based genetic algorithm for solving the 3D bin packing problem (Zhang et al., 2024)](https://doi.org/10.1038/s41598-024-56699-7)
- **Benchmark Instance Generation**: Software tools for creating realistic synthetic datasets for BPP benchmarks.
    - [Reference: Benchmark dataset and instance generator for Real-World 3D-BPP (Zhao et al., 2022)](https://arxiv.org/abs/2208.10641)

## 6. Training Dataset Standards (80/20 Split)
**Related Code Component:** `train_models.py` (`VAL_SPLIT = 0.2`)

Our system follows the industry-standard training protocols for Neural Combinatorial Optimization.

- **Standardized 80/20 Evaluation**: Zhao et al. established the 80/20 train/test configuration as the benchmark for measuring generalization in Deep Reinforcement Learning for packing.
    - [Reference: Online 3D Bin Packing with Constrained Deep Reinforcement Learning (Zhao et al., 2020)](https://arxiv.org/abs/2006.14978)
- **BED-BPP Benchmarking**: The BED-BPP framework formalizes the 80% training and 20% validation split for robotic bin packing datasets.
    - [Reference: BED-BPP: Benchmarking dataset for robotic bin packing problems (Kagerer et al., 2023)](https://floriankagerer.github.io/dataset/)
- **Large-Scale Robotic Packing RL**: Omey Manyar's dissertation provides a comprehensive analysis of the 80/20 split methodology for augmented datasets in robotic bin packing.
    - [Reference: Deep Reinforcement Learning for Robotic Bin Packing (Manyar, 2023)](https://omey-manyar.com/uploads/Ph_D_Dissertation_Omey_Manyar.pdf#:~:text=The%20dataset%20is%20split%2080:20)

## 7. Access Frequency and Operational Efficiency
**Related Code Component:** `optimizer.py` (Accessibility scores)

- **Access-Aware Bin Packing (3D-BPPIF)**: Integrating shape flexibility and ABC analysis for storage assignment.
    - [Reference: A 3D-BPP with item-fragmentation and its application in storage location assignment (Salamati-Hormozi et al., 2024)](https://doi.org/10.1007/s10288-024-00576-6)
- **The Price of Clustering (PoC)**: Theoretical bounds on the number of bins required when items are segregated by cluster.
    - [Reference: The Price of Clustering in Bin Packing (Azar et al., 2019)](https://arxiv.org/abs/1908.06727)
- **Milkrun Logistics Optimization**: Class-based storage policy optimization for picking effort reduction.
    - [Reference: Optimizing the storage assignment in a warehouse served by milkrun logistics (Kovács, 2011)](https://doi.org/10.1016/j.ijpe.2009.10.028)

## 7. Extremal Optimization (EO)
**Related Code Component:** `optimizer.py` (`ExtremalOptimization`)

- **Hybrid Extremal Optimization for BPP**: Combining EO with local search for complex solution spaces.
    - [Reference: A Hybrid Extremal Optimisation Approach for the Bin Packing Problem (Gómez-Meneses, 2009)](https://doi.org/10.1007/978-3-642-02319-4_1)
- **PPO-Driven Hyper-Heuristics for EO**: Applying PPO to select low-level heuristics dynamically in combinatorial optimization.
    - [Reference: A Hyper-Heuristic Algorithm via Proximal Policy Optimization (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0957417424017962)

## 8. Warehouse Picking TSP (Picker Path)
**Related Code Component:** `script.js` (`generatePickerPath`)

- **TSP Heuristics for Order Picking**: Utilizing TSP metaheuristics for routing pickers in warehouses.
    - [Reference: Using a TSP-heuristic for routing order pickers in warehouses (Theys et al., 2010)](https://doi.org/10.1016/j.eswa.2010.02.046)
- **Routing Optimization in Warehouses**: Comparison of VR-based approaches for order picking efficiency.
    - [Reference: Route optimization for warehouse order picking operations via vehicle routing (Shetty et al., 2020)](https://doi.org/10.1007/s42452-020-2114-1)
- **Joint Order Batching and Routing**: Models picker routing as TSP-variants over block-layout warehouses.
    - [Reference: Joint Order Batching, Picker Routing and Sequencing Problem with Deadlines (Cattaruzza et al., 2023)](https://arxiv.org/abs/2303.17834)
- **Coupling Packing Density with Routing**: Studies how packing density affects picker path length in block-layout warehouses.
    - [Reference: ORCA – Online Research @ Cardiff (2024)](https://orca.cardiff.ac.uk/id/eprint/161992/4/Manuscript%20final.pdf)

## 10. Heuristic Repair and Gravity Placement
**Related Code Component:** `optimizer.py` (`repair_solution_compact`)

- **Gravity-Drop Principles**: Heuristic strategies mimicking physical settlement for stable placements.
    - [Reference: A Gravity-Drop Heuristic for Irregular Object Packing (2017)](https://doi.org/10.1109/ICMRE.2017.7924552)
- **Placement Heuristics for 3D-BPP**: Foundational algorithms like Bottom-Left-Front (BLF) and its variants.
    - [Reference: The Three-Dimensional Bin Packing Problem (Martello et al., 2000)](https://doi.org/10.1287/opre.48.2.177.12398)
- **General Literature Review**: A review of various bin packing problem variants and solution strategies.
    - [Reference: Deep Study on the Application of Machine Learning in Bin Packing Problems (2021)](https://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1405-55462021000400002)
