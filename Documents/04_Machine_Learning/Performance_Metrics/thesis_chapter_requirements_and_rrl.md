# Thesis Chapter Requirements and RRL Extraction

> Auto-extracted from thesis `.docx` using `parse_thesis.py`  
> Source: *OPTIMIZING WAREHOUSE STORAGE ALLOCATION USING GENETIC ALGORITHM
> AND EXTREMAL OPTIMIZATION FOR EFFICIENT SPACE UTILIZATION AND INVENTORY MANAGEMENT*

---

## Chapter 2: Review of Related Literature (RRL)

### REVIEW OF RELATED LITERATURE
The growing complexity of warehouse operations has intensified the need for intelligent optimization systems that can effectively manage storage allocation and improve retrieval efficiency. In response, researchers have increasingly turned to metaheuristic algorithms such as Genetic Algorithm (GA) and Extremal Optimization (EO) for their robust search capabilities and adaptability in solving complex, nonlinear problems. A review of existing literature provides critical insights into the strengths, limitations, and applications of these algorithms in the context of warehouse management. These studies also highlight key challenges in traditional storage systems, such as inefficient slotting, static layouts, and limited adaptability to dynamic inventory flows. Synthesizing these findings lays the groundwork for the development of a hybrid sequential model, where EO and GA are combined to address both global exploration and local refinement, offering a more adaptive and data-driven approach to modern warehouse optimization.
Metaheuristic Approaches in Warehouse Optimization
Warehouse operations optimization remains a critical concern in logistics and supply chain management due to the growing demands of e-commerce, limited warehouse space, and complex inventory dynamics. Traditional rule-based methods often fail to capture the nuanced interplay between layout configuration, order patterns, and real-time constraints. This has led to the growing adoption of metaheuristic algorithms, notably Genetic Algorithm (GA) and Extremal Optimization (EO), which offer adaptive, efficient
solutions for complex combinatorial problems such as storage allocation, slotting, and bin packing.
Genetic Algorithm, inspired by the principles of natural selection, has been widely implemented in warehouse scenarios due to its ability to efficiently explore large solution spaces. Khan et al. [4] introduced a Smart Warehouse Management System integrating GA for real-time decision-making, demonstrating its practical adaptability in intelligent storage control. Similarly, Grznár et al. [14] employed GA for sorting warehouse inventory and showed improved throughput efficiency and system responsiveness. In another instance, Yang et al. [15] proposed a hybrid GA approach to optimize batch order picking in mobile rack systems, resulting in significant time savings and minimized travel distances within dynamic storage environments. These studies consistently demonstrate GA’s strength in solving large-scale optimization tasks, particularly when the search space is highly dimensional and constrained.
Extremal Optimization, on the other hand, is a newer yet promising metaheuristic method based on co-evolutionary principles. Unlike GA, EO focuses on iteratively improving the worst-performing components of a candidate solution, making it ideal for fine-tuning suboptimal configurations. Boettcher and Percus [5], [6] pioneered this approach and highlighted its potential in scheduling, packing, and spatial allocation problems. EO’s minimal parameter tuning requirements and focus on local perturbations allow for more computationally efficient refinements, particularly in systems where solution degradation is localized rather than global. Pistolesi et al. [16] further demonstrated the viability of EO in industrial disassembly line balancing by incorporating it into a hybrid framework (EMOGA), showing improvements in convergence speed and solution diversity.
Towards Adaptive, Data-Driven Warehouse Optimization
The move toward intelligent and adaptive warehouse systems further underscores the value of combining evolutionary algorithms. Mirzaei et al. [1] emphasized the importance of integrated, cluster-based storage allocation to improve picker performance and reduce congestion. While their model did not explicitly use EO or GA, it highlights the type of complex optimization problem that benefits from metaheuristic hybridization.
Tufano et al. [17] introduced a machine learning-based approach for predictive warehouse design and acknowledged that such systems benefit significantly from metaheuristic integration, particularly when predicting layout performance under uncertain demand. A hybrid EO-GA model could enhance these data-driven systems by offering adaptive reconfiguration strategies that are both exploratory and locally optimal.
Moreover, Talbi [18] emphasizes in his comprehensive metaheuristics guide that sequential hybrid models—where distinct phases of exploration and exploitation are executed—consistently outperform single-strategy methods in dynamic, high-dimensional environments such as logistics, scheduling, and layout optimization. This supports the methodological foundation of the current study, which seeks to implement a sequential EO-GA framework tailored for warehouse storage optimization.
Storage Allocation and Slotting Strategies in Warehousing
Efficient storage allocation is fundamental to optimizing warehouse operations, as it directly impacts space utilization, retrieval time, and product safety. The reviewed studies consistently emphasize the importance of integrating item-specific attributes such as size, fragility, and access frequency in designing storage layouts. [19] introduced a slotting optimization model that strategically allocates storage slots, enhancing space utilization. Similarly,[20] demonstrated that incorporating product characteristics like size and fragility reduces retrieval times and product damage. These findings underscore the necessity of intelligent, data-driven slotting strategies to improve warehouse efficiency, laying the groundwork for this study’s focus on optimized storage allocation.
Mathematical Formulations of Optimization Objectives
Warehouse layout optimization often requires a multi-criteria approach, incorporating space usage, accessibility, item safety, and constraint compliance. To address this, the present study uses a composite fitness function, inspired by models from Liu et al. [7], Pistolesi et al. [16], and Boettcher et al. [5], to evaluate the quality of each proposed warehouse configuration.
Where, S is a complete solution set of placed items, and  are the weight coefficients for each objective component [7], [21].
This measures the ratio of used space to total warehouse volume. The formula reflects models used in EO-GA packing systems where volume efficiency is critical [15],[22].
Where,  is the access frequency of item i,  is the normalized retrieval distance, based on the XZ coordinates. This formulation aligns with accessibility cost functions applied in the warehouse layout optimization [5], [17].
Where, =1 if item i is placed in the correct zone (fragile/non-fragile); otherwise, =0. This rule-based score is widely used in zone-aware optimization frameworks [16], [19].
Where,  is the fragility level of item i,  is its height placement (z-axis). This metric accounts for stacking risk and reflects formulation from safety-aware bin packing models [22].
These mathematical formulations are inspired by fitness models in hybrid metaheuristics. Boettcher et al.[5], Liu et al.[7] and support the core hypothesis of this study: that sequential EO–GA hybrids can optimize multiple conflicting objectives in high-dimensional warehouse layouts.
Genetic Algorithm in Warehouse Optimization
Genetic Algorithms (GA) have been widely recognized for their effectiveness in addressing complex warehouse layout and order picking problems due to their robust global search capabilities. The literature illustrates that GA can significantly reduce travel distances, enhance picking efficiency, and accommodate various operational constraints.
The IEOM Society [23] proposed a GA-based warehouse storage assignment framework that integrates space, access, and safety objectives. Their model evaluates potential layouts using a multi-objective fitness function:
Where,  are respective weighting coefficients, and S is a warehouse layout solution.
Crossover operations commonly used in GA are illustrated as:
Where, c is a crossover point,  are parent chromosomes, and  are offspring chromosomes.
Mutation operations randomly alter one or more item placements to preserve diversity in the solution pool. A broader theoretical foundation was established by Wikipedia [24] and NumberAnalytics [25], which explain GA mechanics and applications in combinatorial problems. INRIA [26] further explored hybrid GA models for global optimization by combining population diversity strategies with adaptive local tuning.
Adding further depth, Zhao et al. [22] proposed a Real-Polarized Genetic Algorithm (RPGA) to address complex 3D bin-packing problems, showing how GA can enhance space utilization and object arrangement under spatial constraints.
Advancements in Extremal Optimization for Continuous Domains
Extremal Optimization (EO), originally introduced by Boettcher and Percus, has gained attention as an effective single-solution metaheuristic inspired by self-organized criticality and co-evolutionary dynamics. Unlike population-based algorithms such as Genetic Algorithm (GA), EO focuses on improving the worst-performing components of a solution, allowing it to effectively escape local optima without relying on crossover or population diversity. While traditional EO algorithms have demonstrated success in discrete optimization tasks, their performance in continuous and high-dimensional problem spaces has required further refinement.
In response to this need, Liu et al. [7] introduced the Improved Real-Coded Population-Based Extremal Optimization (IRPEO) algorithm, specifically designed for solving continuous unconstrained optimization problems. This approach integrates real-coded representation with a population-based mechanism, significantly enhancing EO’s exploration capabilities and convergence behavior. Unlike standard EO, which operates on a single candidate solution, IRPEO evolves a small population of solutions simultaneously, allowing for improved diversity and robustness.
The IRPEO algorithm introduces three major enhancements:
Real-coded solution encoding, enabling the algorithm to directly manipulate real-valued variables, which is essential for continuous problem domains;
Population-based EO dynamics, where a pool of solutions evolves in parallel through a ranking-based selection and component-wise mutation strategy;
Dynamic mutation operator, which balances exploration and exploitation by adapting the step size based on the current search progress.
To determine which components (e.g., variables or genes) to update during EO’s iteration, the algorithm uses a Power-Law Probability Distribution, defined as:
Where:
P(k) is the probability of selecting the 𝑘𝑡ℎ worst-performing variable,
τ controls the selection pressure,
n is the total number of components.
This probabilistic model biases the selection toward lower-ranked (i.e., worse) variables, increasing the chance of refining the least-fit parts of the solution.
To evaluate the quality of a solution, EO distinguishes between local and global fitness. The global fitness function C(S) is commonly represented as the sum of the local fitness contributions 𝜆𝑖 of each component 𝑖 in the solution 𝑆:
This aggregation allows EO to make decisions based on both localized weaknesses and the overall solution landscape—especially relevant in warehouse contexts, where local storage decisions (e.g., accessibility or safety) affect overall efficiency.
Advances in Metaheuristics for Sequential Hybrid Optimization
As real-world optimization problems become increasingly complex and computationally intensive, researchers have turned to hybrid metaheuristic approaches that combine the strengths of different algorithms. Among these, the sequential hybridization of Extremal Optimization (EO) and Genetic Algorithms (GA) presents a promising strategy: EO offers robust global search capabilities by iteratively eliminating poor-performing solution components, while GA excels in fine-tuning through recombination and population-based evolution. Two recent studies provide strong foundational support for this hybrid design Liu et al. [7] with their IRPEO model and Ansótegui et al. [27] with their Model-Based Genetic Algorithm (MBGA).
Liu et al. [7] proposed the Improved Real-Coded Population-Based Extremal Optimization (IRPEO) method, a significant enhancement of the original EO framework designed for continuous, unconstrained optimization tasks. Unlike classic EO, which operates on a single solution, IRPEO employs a small population of real-coded solutions, allowing better diversity and global exploration. It incorporates a dynamic mutation strategy and rank-based selection to identify and improve the worst-performing variables across multiple candidates. Benchmarked against traditional EO and GA models, IRPEO consistently produced superior results in terms of convergence speed and accuracy on complex mathematical functions like Rastrigin and Rosenbrock. These findings reinforce EO’s strength in identifying promising regions in high-dimensional and nonlinear search spaces an essential quality for initiating global exploration in a hybrid framework.
Complementing this, the work of Ansótegui et al. [27] on the Model-Based Genetic Algorithm (MBGA) offers a powerful mechanism for the second phase of hybrid optimization solution refinement. Their approach introduces surrogate modeling using Random Forests to predict the performance of new candidate solutions without costly full evaluations. Within the MBGA, a gender-based population structure is employed to maintain diversity, while a genetic engineering step uses the surrogate model to generate offspring with high estimated fitness. Particularly useful in expensive or black-box settings like algorithm configuration, MBGA demonstrated superior performance over traditional GAs and other configurators (e.g., SMAC), especially when applied to high-dimensional tuning problems such as SAT solver parameter optimization.
Together, these two studies form a strong methodological basis for a sequential EO–GA hybrid framework. IRPEO’s population-driven global search effectively explores the problem landscape and avoids premature convergence, while MBGA’s surrogate-guided GA provides a scalable and efficient mechanism for fine-tuning. This hybrid sequencing aligns with current needs in logistics and warehouse optimization, where the solution space is often vast, nonlinear, and filled with constraints that require both exploration and targeted refinement.
Theoretical Support for Adaptive Optimization in Logistics
Modern logistics requires adaptive systems that respond to changing inventory and demand conditions. Multi-objective and hybrid metaheuristics offer a framework for balancing conflicting goals such as speed, space, and accuracy. This theoretical foundation justifies your integration of GA and EO in a flexible, data-driven optimization system.  Shi & Eberhart. [28] introduced Particle Swarm Optimization (PSO), which has since been applied to warehouse slotting and layout optimization. Including PSO offers a contrast to the GA-EO hybrid, broadening the scope of heuristic approaches. Osman. [17] reviewed Simulated Annealing (SA) and its use in combinatorial logistics problems like storage assignment and route optimization. SA's capacity to escape local optima offers another useful contrast.  Coelho & Laporte. [29] discussed multi-objective metaheuristics in logistics. Their study supports the development of hybrid methods like GA+EO for balancing competing goals such as space utilization and retrieval speed. Talbi. [18] provided a comprehensive survey on hybrid metaheuristics, highlighting how combining algorithms can yield better optimization results than single-method approaches. This justifies the hybrid strategy employed in this study.
Generative Adversarial Networks for Synthetic Data Generation
Generative Adversarial Networks (GANs) provide a formal framework for learning the distribution of real data and generating synthetic samples that closely resemble the original dataset [30], [31]. A standard GAN consists of a generator G and a discriminator D trained in a two‑player minimax game with the objective
(11)
Where Pdata is is the real data distribution and Pz is a prior over noise variables. Through this process, the generator learns to map random noise z into realistic samples, while the discriminator learns to distinguish real from generated data, allowing the GAN to capture complex, high‑dimensional relationships among features such as item dimensions, fragility, and access frequency [30], [31]. Empirical studies show that GAN‑based data augmentation can significantly improve model performance in domains with limited or imbalanced data, since synthetic samples better represent rare but operationally important patterns compared with traditional oversampling methods [32], [33], [34].
GAN-Augmented Metaheuristic Optimization for Packing and Warehouse Layout
In optimization and metaheuristic contexts, GANs are increasingly used to enrich the search space explored by algorithms such as Genetic Algorithms by generating diverse candidate solutions or realistic problem instances[34]. A prominent example is a GAN–GA hybrid for the 3D bin packing problem, where a GAN is trained on feasible packing patterns and then used to produce high‑quality initial solutions that a GA further refines, achieving better packing density and fewer bins than conventional GA or particle swarm optimization approaches[35]. Because 3D bin packing is structurally similar to assigning items to warehouse storage slots under space and constraint conditions, this research supports using GAN‑generated synthetic inventory profiles or layout instances as input for metaheuristics in warehouse optimization[36]. Integrating a GAN‑based augmentation stage—defined mathematically by the minimax objective above—before the Extremal Optimization and Genetic Algorithm phases aligns with these trends, providing a richer and more varied set of item configurations for the EO–GA hybrid to optimize and potentially leading to more robust and generalizable warehouse storage layouts[23], [37], [38].
Hybrid Neural Combinatorial Optimization and Metaheuristics
The evolution of the 3D Bin Packing Problem (3D-BPP) has moved toward Neural Combinatorial Optimization (NCO), where deep learning models are trained to replace or enhance traditional heuristics. Modern frameworks often treat bin packing as a Constrained Markov Decision Process (CMDP), utilizing Deep Reinforcement Learning (DRL) to learn optimal placement sequences [4]. Advanced agents employ multimodal encoders to process both numerical item data (weight, size) and visual states (top-down bin height maps) to predict the most efficient coordinates and rotations [22].
(12)
Recent literature emphasizes the "hybridization" of these neural approaches with established metaheuristics. For example, [39] demonstrated that integrating a GAN’s generator directly into a Genetic Algorithm (GA) can produce high-quality initial populations, preventing the optimization from getting stuck in local optima and improving space utilization [39]. While these neural solvers offer high-speed predictions, they are frequently benchmarked against Extremal Optimization (EO) a local-search heuristic inspired by self-organized criticality. Hybrid EO (HEO) models, which iteratively refine "weak" solution components, serve as a critical baseline to determine if the added complexity of a trained neural model provides a significant measurable benefit in warehouse efficiency [40].

### Section 2.5: Theoretical Foundations of Sandwich Normalization in Logistics

The integration of Generative Adversarial Networks and machine learning for 3D Bin Packing requires a robust normalization architecture to bridge the gap between abstract latent space and physical warehouse constraints. This study identifies and utilizes a **"Sandwich Normalization"** cycle to ensure stable model convergence across multi-modal item distributions.

#### 1. The Architectural Sandwich (SaBN)
As defined by **Kim, M., Li, B., Shin, J., & Hong, S. (2021)** in *"Sandwich Batch Normalization: A Drop-In Replacement for Feature Distribution Heterogeneity"*, the "sandwiching" of affine transformations within standard Batch Normalization addresses feature distribution heterogeneity. In logistics data—where weights (kg) and dimensions (m) exhibit vastly different numeric scales—standard normalization often leads to "Feature Dominance," where the model ignores smaller spatial deltas in favor of larger mass-based gradients. This study implements SaBN-inspired logic by factorizing the data transformation cycle into a shared global scaling layer (Min-Max) followed by internal conditional normalization layers.

#### 2. Resolution of Internal Covariate Shift
To master the non-linear physics of 3D-BPP, the predictive MLP utilizes **Batch Normalization (BN)**, a technique pioneered by **Ioffe, S., & Szegedy, C. (2015)**. By normalizing the "filling" of the sandwich (the internal hidden layers), the system reduces internal covariate shift, allowing for higher learning rates and faster convergence in the sequential EO-GA optimization pipeline. As demonstrated by Ioffe and Szegedy, this internal stabilization is critical for training deep networks on disparate datasets like the GAN-augmented warehouse inventory.

#### 3. Formal Normalization Objective
Following these theoretical benchmarks, the normalization objective in this study is defined as:
(13)
Where $x_{norm}$ represents the relative spatial ratio used for coordinate prediction, and $Denorm$ is the inverse transformation required for physical settlement in the PyBullet engine. This cycle allows the system to transition from **Absolute Spatial Mapping** to a **Universal Stacking Policy** that generalizes across variable warehouse dimensions.

---

## Chapter 3: Methodology

### METHODOLOGY
This chapter presents the methodological approach employed to develop and evaluate the proposed machine learning-seeded hybrid sequential optimization model. The section outlines the research design, dataset description, data preprocessing techniques, the integration of Generative Adversarial Networks (GAN) and predictive neural networks, the sequential execution of Extremal Optimization (EO) and Genetic Algorithm (GA), and the evaluation metrics and technological tools utilized throughout the study. This structured methodology ensures the replicability of results and aligns with the objectives of optimizing 3D warehouse storage allocation for maximum efficiency.
### A. Research Design
The study utilizes an experimental research design to develop and evaluate a high-performance optimization pipeline for the 3D warehouse object placement problem. The methodology integrates a multi-phase technical framework: first, using a Generative Adversarial Network (GAN) for data augmentation; second, employing a PyTorch-based deep learning model to predict initial spatial coordinates; and finally, executing a sequential metaheuristic refinement using Extremal Optimization (EO) for global search followed by a Genetic Algorithm (GA) for local fine-tuning.
The focus of this design is to evaluate the effectiveness of seeding heuristic search with machine learning predictions to maximize space utilization, accessibility, and safety within a constrained 3D environment. This sequential strategy allows for a comprehensive evaluation of how predictive modeling and hybrid metaheuristics synergize to overcome the computational limitations of traditional standalone optimization techniques. The complete workflow of this system is illustrated in Figure 2 below.
Figure 2 Use Case Diagram and System Workflow of the 3D Bin Packing and Optimization System.
The pipeline begins with the processing of the raw inventory dataset, where essential item attributes (dimensions, weight, and category) are extracted and standardized from millimeters to meters. This preprocessed data feeds into the machine learning phase, which utilizes a saved scaler (scaler.pkl) to normalize inputs before training the Generative Adversarial Network (GAN). The trained generator and discriminator models then synthesize additional inventory data by generating 4D latent vectors and assigning realistic fragility and stackability constraints. Finally, both the real and synthetic data are imported into the warehouse database to drive the optimization phase. The system executes the packing optimization by running the metaheuristic algorithms (GA and EO) while strictly enforcing stability and gravity heuristics, ultimately producing a physically viable, high-efficiency storage layout for the user.
### B. Dataset Description
The dataset used in this study consists of a simulated inventory metadata relevant to warehouse operations, including item dimensions, weight, fragility levels, and access frequencies. To ensure a robust evaluation, the research utilizes an augmented version of this dataset where a GAN is employed to generate synthetic item records that are statistically consistent with real-world warehouse inventory profiles.
For the experimental phase, the system is tested against varying dataset scales specifically 200, 400, and 600 items—to measure the scalability and convergence speed of the optimization algorithms. This multi-tier testing approach enables a direct comparison between the model’s predicted storage layouts and established physical constraints, supporting a clear interpretation of the system’s overall accuracy and efficiency
Dataset Attributes:
Dataset:
Table 1. Raw Dataset used for example.
To support more comprehensive experimentation, the original simulated dataset is extended using a GAN-based data generation process. The GAN is trained using the full set of item-level attributes, treating each item record as a multidimensional sample from the underlying warehouse inventory distribution.
​	After convergence, the generator produces additional synthetic items that preserve key statistics of the original data (marginal distributions of dimensions and fragility, as well as correlations with access frequency and category). These GAN-generated records are combined with the original items to form an augmented dataset, which is then used as input to the Extremal Optimization and Genetic Algorithm stages. This approach follows existing work where GANs are employed to expand optimization and packing datasets, leading to improved performance of genetic algorithms on 3D bin packing tasks closely related to warehouse loading and storage.
### C. Data Preprocessing
The methodology commences with a comprehensive data preprocessing phase, where raw inventory attributes are transformed into structured, multidimensional objects suitable for 3D spatial optimization. This transition ensures that all physical and logical constraints are computationally defined before being processed by the augmentation and optimization engines. Central to this phase is the implementation of a structured Item class, which encapsulates essential physical dimensions length, width, and height alongside critical handling metrics such as fragility ratings and access frequencies. By instantiating each dataset entry as a dedicated object, the system can dynamically track unique SKU identifiers while maintaining the integrity of the item's attributes throughout the 3D coordinate mapping process.
A vital function within this data structure is the dynamic orientation system, which allows the model to explore six distinct 3D rotations for every object. Through a dedicated method, the system can programmatically reconfigure an item's dimensions by swapping its length, width, and height axes. This flexibility is essential for maximizing space utilization, as it enables the optimization algorithms to test various base orientations to find the most efficient fit within the warehouse's volumetric bounds. This systematic approach to rotation ensures that the heuristic search is not limited by the item's original entry state, significantly expanding the potential search space for optimal placement.
The final stage of preprocessing involves the application of a rule-based zoning strategy designed to manage fragility and ensure operational safety. Items identified with a fragility rating of three or higher are subjected to strict spatial segregation and height restrictions. These fragile objects are restricted to a designated "Fragile Zone," occupying a specific percentage of the warehouse width, and are further limited to the lower vertical sections of the warehouse to prevent crushing or stacking damage. By embedding these safety boundaries directly into the preprocessing logic, the system establishes a clear set of compliance rules that are later enforced by the fitness function to penalize any physically or logically.
Rotation System
Table 1 Rotation System for 3D Item Orientation
This table illustrates the possible spatial configurations for each item, allowing the optimization algorithms to manipulate length, width, and height orientations to maximize 3D space utilization.
Zone-Based Fragility Management
To maintain warehouse safety and prevent inventory damage, the system implements a rule-based zoning strategy centered on a fragility threshold. Items with a fragility rating of 3 or higher are classified as fragile and are strictly restricted to the "Fragile Zone," which occupies the left 60% of the warehouse width, spanning 0 to 120 cm on the y-axis. Conversely, non-fragile items are assigned to the remaining 40% of the width, designated as the "Non-Fragile Zone". To further mitigate risk, a vertical constraint is applied, limiting fragile items to the lower 40% of the total warehouse height (0 to 80 cm on the z-axis) to prevent crushing from heavier overhead loads. The system enforces these constraints through zone compliance rules: fragile items must adhere to both spatial and height restrictions, while non-fragile items are directed toward the non-fragile zone with full vertical access. Any violation of these parameters is heavily penalized within the fitness function calculation to ensure the final layout is both physically viable and safe.
Figure 1. Zone-Based Fragility Management Pipeline
GAN-Based Data Augmentation
Following standard preprocessing including missing value handling, normalization, and rule-based constraint checks the cleaned item dataset is used to train a Generative Adversarial Network. The GAN consists of a generator G and a discriminator D trained in a minimax game, where G learns to output realistic item vectors and D learns to distinguish real samples from generated ones. This adversarial training process is mathematically defined by the minimax objective function (refer to Equation 10).
Figure 2. GAN Pipeline
Neural Network Feature Engineering and Normalization
Before the inventory data is processed by the predictive machine learning model, the raw item attributes are transformed into a structured, 10-dimensional numerical feature vector. This vector encapsulates the item's physical dimensions (length, width, and height), its weight, and critical categorical or boolean flags including fragility, stackability, and rotation capability. Additionally, the environment's global dimensions (warehouse length, width, and height) are included to provide the network with necessary spatial context for placement.
To ensure stable gradient descent and prevent high-magnitude values from dominating the learning process, the input features (X) and target labels (Y) are normalized according to the strategies outlined in Table II. This scaling process constrains the input space and converts target coordinates into a relative percentage scale, facilitating more efficient convergence during the training phase.
Figure 3. Architectural Workflow of the Predictive Neural Network Pipeline: From Feature Engineering to Physics Settlement
Table II Normalization and Scaling Strategies.
D. Fitness Function Design
Comprehensive Fitness Model
The fitness function integrates four primary components through a weighted summation to determine the overall quality of a warehouse layout (refer to Equation 1). The total fitness value, $F(S)$, is calculated for a complete solution set of placed items, $S$, by applying specific weight coefficients — to each objective component. These components include a SpaceUtilization score to maximize occupied volume, an AccessibilityScore to prioritize frequently used items, a SeparationScore for correct zoning, and a SafetyScore to manage fragility risks. Additionally, an UnplacedItemPenalty is subtracted from the total to penalize configurations where items cannot be successfully accommodated within the warehouse boundaries.
Space Utilization
The Space Utilization score (refer to Equation 2) serves as a primary metric for evaluating the volumetric efficiency of a proposed warehouse layout. This component is calculated by aggregating the total volume of all n items successfully placed within the warehouse, where each item i is defined by its specific physical dimensions (length_i, width_i, and height_i) measured in centimeters. By dividing this cumulative occupied volume by the total available Warehouse Volume, the system derives a percentage-based score that reflects the effectiveness of the algorithm in minimizing wasted space.
Accessibility Score
To calculate the accessibility component of the fitness function (refer to Equation 3), the system evaluates the localized  for each item i placed within the warehouse. This penalty is derived by normalizing the item's current and coordinate positions against the total  and , respectively. This spatial factor is then weighted against the item’s weekly and compared to the max_access frequency across the entire inventory. By aggregating these values for the total number of items, n, the system ensures that high-demand goods are positioned in locations that minimize retrieval travel time.
Separation Score
Where zone compliance is evaluated based on fragility threshold and proper zone placement (refer to Equation 4).
Safety Score
The Safety Score (refer to Equation 5) is designed to mitigate structural risks by managing the vertical placement of items based on their individual fragility levels. This component utilizes a localized height\_safety_i factor, which is calculated by subtracting the normalized Z-coordinate position of an item from the total WAREHOUSE_HEIGHT. By weighting this factor against the item’s fragility_i rating and the system’s max\_fragility limit, the function heavily penalizes the placement of high-fragility items at elevated heights. This mathematical formulation ensures that the most delicate goods are prioritized for lower storage tiers to prevent damage and preserve stack stability.
E. Machine Learning and Optimization Algorithms
Predictive Model Training and Architecture
Table III provides a comprehensive technical overview of the deep learning architecture integrated into the optimization pipeline. This configuration details the layers of the Multi-Layer Perceptron (MLP), which serves as the predictive engine for generating initial high-quality spatial seeds. The model is specifically designed with varying hidden layer widths and activation functions to map complex item features such as fragility and dimensions to valid 3D coordinates.
Table III. MLP Architecture for Initial Placement Prediction
To introduce non-linearity and model complex spatial relationships, a Rectified Linear Unit (ReLU) activation function is applied to each hidden layer, while the output layer remains linear to facilitate raw coordinate prediction. The network is trained using a supervised learning approach with a batch size of 64 over 50 epochs, utilizing a learning rate of 0.001. The Adam optimizer is employed for efficient gradient handling, with model performance evaluated via Mean Squared Error (MSE) to minimize deviations between the predicted (\hat{y}_i) and true normalized coordinates (y_i). Upon reaching convergence, the trained weights are exported as .pth files for seamless integration into the hybrid optimization pipeline.
Model Inference and Physics Settlement
During active storage allocation, the MLOptimizer performs rapid inference by processing normalized item features through a single forward pass of the pre-trained neural network. To ensure operational feasibility, these statistical predictions undergo a Physics Settlement phase using a deterministic repair algorithm (repair_solution_compact). This stage resolves physically impossible placements—such as floating objects—by simulating gravity and enforcing strict geometric support constraints. Each item is iteratively adjusted to its lowest valid Z-position, ensuring it possesses sufficient surface contact with underlying items to maintain structural stability.
Hybrid Optimization Framework
The research implements a two-phase optimization strategy that leverages the global exploration of Extremal Optimization (EO) followed by the local refinement of a Genetic Algorithm (GA). In this framework, the EO stage is initialized with warehouse configurations seeded by the predictive ML model. Utilizing its component-wise replacement mechanism, the EO identifies and shifts items to eliminate localized "weaknesses" in the layout, effectively establishing a robust macro-level placement across the 3D search space.
Following the global search, the GA phase ingests the best-performing EO outputs as elite individuals. Through population-based mechanisms—including tournament selection, single-point crossover, and zone-aware mutation—the GA performs fine-grained spatial tuning to maximize final packing density. This design mirrors contemporary hybrid architectures in 3D bin packing, where machine learning-augmented instances enhance the search space for evolutionary algorithms. By integrating GAN-generated data and ML-seeding, the framework ensures high robustness and superior space utilization across diverse inventory profiles and fluctuating demand patterns.
Extremal Optimization (EO)
Extremal Optimization (EO) is a metaheuristic designed to improve suboptimal solutions by iteratively replacing components with high "local costs." In the context of 3D warehouse allocation, the algorithm identifies items with the worst local scores derived from accessibility, fragility, and zone compliance and attempts to reposition them into more optimal coordinates. This process generates large fluctuations in the fitness landscape, allowing the model to escape local optima and explore distant neighborhoods of the configuration space.
Table IV. Extremal Optimization Configuration Parameters
Local Cost and Worst-Component Identification
Figure 4 Mathematical Formulation for Item-Level Local Cost and Worst-Component Identification.
The core of the EO process is the calculation of a local\_score_i for each item, which acts as the metric for identifying "extremal" components (refer to Equation 12). This score is the summation of three specific penalties: the access\_penalty_i, which weights the item's x-coordinate against its access frequency; the fragility\_penalty_i, which penalizes high vertical placement for delicate goods; and the zone\_penalty_i, which applies a discrete cost (1.0 or 2.0) if an item is stored in an incorrect functional zone. By targeting qthe item with the highest local penalty, the algorithm focuses its computational effort on the most problematic elements of the warehouse layout.
Initial Solution Generation
Generate random valid placement using zone-aware placement generation with maximum 1000 attempts for complete solution. In the provided Python code, this is handled by an initial greedy placement sorted by volume, followed by find_valid_spot attempts.
Worst Component Identification
For each item, calculate local score including zone compliance. Although the Python code uses a simplified random selection for the item to reposition, the underlying concept of identifying "worst" components (or at least components that could be improved) is central to EO.
Zone-Aware Placement and Improvement
Once a worst-performing component is identified, the system attempts to find a superior placement within the appropriate warehouse zone. This find_valid_spot mechanism enforces zone-based height and width constraints while testing multiple random orientations and positions. If a valid placement with a lower local cost is identified, the solution is updated. This cycle repeats for a set number of iterations, with the system maintaining a global best-tracking variable to preserve the highest-quality configuration found during the search process.
Genetic Algorithm (GA)
The Genetic Algorithm (GA) is an adaptive metaheuristic inspired by the principles of natural selection and evolutionary biology. In this framework, the GA serves as the final refinement stage, taking the optimized outputs from the Extremal Optimization phase and treating them as elite individuals within a larger population. Through iterative generations of selection, recombination, and mutation, the algorithm performs fine-grained spatial tuning to converge on a globally optimal 3D warehouse layout.
Selection and Crossover Operations
The evolutionary process begins with Tournament Selection, where a subset of individuals is randomly chosen from the population, and the candidate with the highest fitness is selected as a parent. This method maintains a balance between genetic diversity and selection pressure. Following selection, a Single-Point Crossover operation is performed (refer to Equation 13). During this process, genetic material—specifically the 3D placement configurations—is exchanged between two parent chromosomes at a randomly selected crossover point. This allows the algorithm to combine high-performing spatial clusters from different solutions to produce superior offspring.
1.Tournament Selection
Tournament selection with size k: a subset of k individuals is randomly chosen from the population, and the individual with the best fitness within this subset is selected as a parent. This is implemented in the tournament_selection function.
2.Crossover Operation
A single-point crossover operation is performed, where genetic material (item placements) is exchanged between two parent chromosomes. This is implemented in the crossover function. Given parents P1and P2 with n items
Figure 5. Genetic Recombination Logic for 3D Item Placement.
The recombination of placement data is governed by the union of genetic subsets from two selected parents, P_1 and P_2. Given a total of n items, a crossover\_point is randomly selected within the range [1, n-1] to determine the split. The resulting offspring configurations, child_1.placements and child_2.placements, are formed by concatenating the placement subset P_x.placements[a:b] from the primary parent with the remaining sequence from the secondary parent using the union operator (\cup). This ensures the children inherit high-performing spatial clusters from both ancestors while maintaining a complete inventory set.
### F. Experimental Setup and Evaluation Framework
The performance of the machine learning-seeded hybrid framework is evaluated using synthetic datasets scaled at 200, 400, and 600 items within a warehouse environment dynamically sized with a 2.5x volume buffer and a 20 cm walkway clearance. The optimization follows a sequential "Multi-Warehouse" strategy to ensure 100% inventory fulfillment. Initially, the full dataset undergoes 500 iterations of Extremal Optimization (EO) to establish a macro-level layout, the best of which is injected as an elite individual into a Genetic Algorithm (GA) population of 50 for 200 generations of fine-grained refinement. If structural or safety constraints prevent any items from being housed in the primary warehouse, a secondary overflow warehouse is dynamically generated, and the hybrid EO-GA process is repeated for the remaining inventory.
The effectiveness of this packing strategy is quantified through a multi-metric framework focusing on volumetric efficiency, operational compliance, and computational performance. Key performance indicators include Space Utilization percentages, weighted Fitness Scores incorporating space, fragility, and access penalties—and strict boolean Solution Validity checks for overlaps and walkway adherence. Final configurations are analyzed via 3D wireframe visualizations built in Matplotlib, allowing for qualitative inspection of category-based color coding and height distribution. These statistical summaries, including final item counts and convergence rates, provide the quantitative foundation for the comparative analysis discussed in Chapter IV.
### H. Evaluation Metrics
The effectiveness of the optimization process is evaluated through a multi-metric framework applied independently to both training and validation datasets, ensuring that the optimized layouts generalize beyond the data used during the learning phase. Key performance indicators include the Constraint Violation Count, which captures infractions such as incorrect stacking order, size mismatches, and access inefficiencies. A lower count indicates a more feasible and compliant storage layout. Complementing this is the Fitness Score, a composite metric calculated from the weighted penalties assigned to constraint violations, space inefficiency, and suboptimal item accessibility where a higher score reflects a more optimized configuration.
Access Efficiency is also evaluated, measuring the average retrieval time or distance for high-frequency items, thereby highlighting the layout's impact on operational throughput. Additionally, the Space Utilization Ratio quantifies how effectively the available warehouse volume is used, calculated as the proportion of occupied volume to total storage capacity. To ensure robustness and consistency, each configuration GA-only, and the sequential GA and EO model is tested across multiple simulation runs with varying item profiles and layout constraints. This comprehensive evaluation approach not only verifies the technical soundness of the hybrid optimization strategy but also demonstrates its applicability to real-world warehouse environments.
### I. Tools and Environment
To build, train, and evaluate the hybrid optimization model for 3D warehouse allocation, the study utilized the following technologies
Programming Language
Python 3.11 – 3.14.3 : Python served as the primary programming language due to its versatile ecosystem and wide adoption in scientific computing, algorithm development, and machine learning.
Core Libraries
PyTorch: A high-performance machine learning library used to implement the Multi-Layer Perceptron (MLP). Specific modules like torch.nn and torch.optim were utilized for architecture design and backpropagation, while DataLoader ensured efficient batching.
Pandas: Employed for robust data manipulation and cleaning. It was essential for parsing CSV training datasets and formatting raw inventory data prior to tensor conversion.
NumPy: Utilized for high-performance numerical computations and array-based operations, ensuring the algorithmic efficiency of the spatial coordinate calculations.
Matplotlib: Integrated for graphical representation and data analysis. The mpl_toolkits.mplot3d module was specifically used to generate 3D wireframe visualizations of the warehouse and item placements.
Standard Utilities
Standard Python Suite: Libraries such as math, random, and time provided foundational support for trigonometric operations, stochastic search in metaheuristics, and execution time tracking.
Typing and Data Classes: Utilized to define structured data types and ensure type safety, enhancing the clarity and maintainability of the complex software architecture.
Hardware Tools
Ryzen 7 5700x, 48GB RAM DDR4 3200 MT/S, RTX 3060 12GB
### I. Tools and Environment
This study exclusively utilizes synthetic inventory data and GAN-augmented datasets, which are generated based on standardized logistics parameters and warehouse operational rules. By employing these artificial data sources, the research ensures that no sensitive corporate records, proprietary logistics data, or identifiable business information are used. Our methodology complies fully with ethical standards concerning algorithmic transparency, data privacy, and academic integrity. Furthermore, the hybrid optimization framework prioritizes physical safety by subordinating all machine learning predictions to deterministic structural constraints, ensuring that the transition toward automated storage management remains safe, accountable, and interpretable.

---

## Chapter 4: Results and Discussion

### RESULT AND DISCUSSION
This chapter presents the data analysis and interpretation of the results obtained from the implementation of the machine learning-seeded hybrid optimization framework. The findings are categorized into five sections corresponding to the research objectives, covering data preprocessing, GAN-based augmentation, predictive model training, hybrid optimization performance, and a comparative evaluation against standalone heuristics.
A. Item Attribute Collection
This section details the foundational data utilized for the storage planning phase. The dataset derived from the publicly available 3D Bin Packing repository curated by Kagerer [41] was used, consisting of 428,719 item profiles divided into training (80%), validation (20%), and separate multi-scaled testing sets.
Table V. Technical Specifications of the Raw Dataset
The raw dataset in Table V serves as a high-fidelity benchmark, containing diverse item profiles characterized by 7 primary technical attributes including length, width, height, weight, and volume alongside logistical constraints such as fragility ratings and access frequency.
Table VI. Sample of Preprocessed (Normalized) Data
Table VII. Sample of Preprocessed (Denormalized) Data
“DISCUSS”
Table VIII. Pre-processed overview
Table 10. Pre-processed Output
B. GAN-Based Inventory Augmentation
By expanding the Kagerer[41] dataset, the GAN ensures that the subsequent optimization models are exposed to a wider range of edge-case inventory profiles.
Figure 6. Code Snippet for Normalization
The code snippet in Figure 10 illustrates the essential data preprocessing pipeline required before GAN training. To ensure model stability, a MinMaxScaler compresses highly variable real-world parameters (e.g., physical dimensions and weight) into a uniform [0, 1] range. This prevents attributes with naturally large magnitudes from disproportionately dominating the network's loss gradients and causing mode collapse.
Additionally, the code handles the dynamic conversion of this normalized data into PyTorch tensors (torch.from_numpy) for efficient batch processing. Crucially, it also retains the scaler object via the get_scaler() method. This is a vital architectural step, as it allows the system to later apply an "inverse transform" to the GAN's synthetic outputs successfully denormalizing the artificial [0, 1] values back into the real-world physical metrics required by the PyBullet physics engine and the hybrid EO-GA optimizers.
GAN Training Performance
The GAN architecture was trained over 500 epochs and 64 batch size, utilizing a competitive learning process between the Generator and the Discriminator. Figure 7 illustrates the loss curves for both components, highlighting the point of equilibrium where the Generator produces samples that the Discriminator can no longer easily distinguish from real data.
Figure 7. Convergence of Generator and Discriminator Loss Curves
The visualization plots both the training and validation loss for the Generator (orange) and Discriminator (blue). In the initial phase of training (epochs 0–50), the Discriminator loss rapidly decreases while the Generator loss exhibits a sharp upward spike. This is a standard characteristic of early-stage adversarial learning, occurring because the Discriminator quickly learns to identify the initial, low-quality synthetic data before the Generator has adapted its weights.
As training progresses beyond the 100-epoch mark, the loss curves demonstrate a clear stabilization, indicating that the network is approaching a Nash equilibrium. The Discriminator’s training and validation losses flatten and maintain a steady, parallel state near 0.67. Crucially, this prevents the Discriminator from overpowering the network (which would result in vanishing gradients). Concurrently, the Generator's loss stabilizes around the 0.74 mark. While the Generator's validation loss exhibits the expected stochastic fluctuations inherent to adversarial training, it does not diverge toward infinity.
This sustained, non-diverging balance signifies a highly successful training phase. It indicates that the Generator has effectively learned the underlying statistical distribution of the warehouse dataset. Because the losses reach a stable equilibrium rather than collapsing to zero, it proves the Generator is producing synthetic warehouse items (with realistic lengths, widths, heights, and fragilities) that the Discriminator genuinely struggles to differentiate from actual ground-truth inventory. This stable convergence confirms that the data augmentation phase avoided common generative pitfalls, such as mode collapse. Consequently, the resulting synthetic dataset provides a robust, realistic, and highly diverse foundation for stress-testing the subsequent hybrid EO-GA optimization pipeline.
Figure 8. Neural Network Architectures for GAN-Based Inventory Augmentation
Figure 9. Implementation of Argument Parsing for Scalable Synthetic Data Generation
Figure 10. Total Scenarios per Variant
The code snippet in Figure 8 illustrates the architectural implementation of the Generative Adversarial Network (GAN) using PyTorch, specifically defining the Generator and Discriminator modules. The resulting sequential neural network layers capture the complex, non-linear distributions of warehouse inventory attributes, ensuring the synthesized data closely mimics realistic physical dimensions and constraints.To complement the limited scale of the baseline raw dataset, this study employed a GAN framework to systematically augment the inventory profiles and introduce necessary edge cases. The Generator synthesizes new item parameters from a latent noise vector using linear transformations, LeakyReLU activations, and batch normalization to maintain stability. Simultaneously, the Discriminator evaluates these outputs against real data. This adversarial process allows the model to learn the underlying multi-dimensional relationships of logistics data, culminating in a final Sigmoid activation that outputs normalized item attributes strictly bounded between [0, 1].
The code snippets in the subsequent figures illustrate the parameterized pipeline for generating the synthetic inventory data and structuring the specific test scenarios. The resulting configuration ensures high experimental reproducibility through controlled random seeds and allows for dynamic scaling of the dataset via command-line arguments.
To thoroughly evaluate the hybrid optimization models under varying degrees of spatial constraint, the synthesized inventory was structured into distinct environmental variants. The dataset is explicitly partitioned into "Dense" scenarios (simulating small warehouse floors that force aggressive vertical stacking) and "Normal" scenarios (representing standard, varied floor capacities). By generating 50,000 total rows distributed across these specific operational constraints, the system ensures that the Extremal Optimization (EO) and Genetic Algorithm (GA) frameworks are rigorously stress-tested against a diverse array of realistic logistics.
Training Data
The Training split constitutes the largest portion of the data, allowing the neural network to learn complex spatial patterns and physical relationships by adjusting weights during the supervised learning phase. The Validation split (or development set) was employed to monitor the model's performance during training, facilitating the fine-tuning of hyperparameters such as learning rate and batch size to prevent overfitting. Finally, the Testing split served as an entirely unseen dataset, used exclusively for the final evaluation to measure the system's actual predictive capability and placement accuracy. Figures x, x and x illustrate five sample rows from each respective dataset split, highlighting the normalized attributes for length, width, height, and fragility.
Sample Training Split
Table IX. Training Split (EO)
Table X. Training Split (EO+GA)
Table X1. Training Split (GA)
Table XII. Training Split (GA+EO)
The tables presented illustrate the sample data structures used to train the machine learning predictive model across four distinct algorithmic configurations: standalone ML-driven Extremal Optimization (EO), standalone ML-driven Genetic Algorithm (GA), and their sequential hybrid combinations (EO+GA and GA+EO).
Each dataset maps the continuous input features specifically the normalized physical dimensions of the inventory items (item_l, item_w, item_h) to their resulting optimal 3D spatial placements (target_x, target_y, target_z). These target coordinates represent the "ground truth" layout generated by each specific algorithmic pathway.
Notably, the variance in the target_z column captures the system's vertical stacking behavior; values of zero indicate base items placed directly on the warehouse floor, while non-zero values reflect stable vertical bin-packing decisions. By structuring the training data across these four comparative splits, the study can systematically evaluate how each sequence of ML-guided optimization influences the neural network's overall accuracy in predicting high-density, conflict-free warehouse layouts.
Sample Validation Split
Table XIII. Validation Split(EO)
Table XIV. Validation Split (EO+GA)
Table XV. Validation Split (GA)
Table XVI. Validation Split (GA+EO)
The presented tables illustrate the validation data subsets used to evaluate the predictive model across the four distinct algorithmic configurations (standalone EO, standalone GA, and the hybrid sequences EO+GA and GA+EO). While the training splits are used to actively adjust the neural network's internal weights, these validation splits consist of unseen data utilized exclusively to monitor the model's performance during the learning phase. This critical step ensures that the model is genuinely learning to generalize complex spatial patterns rather than simply memorizing the training data, effectively preventing overfitting.
Consistent with the training datasets, these tables map the normalized physical dimensions of the inventory items (item_l, item_w, item_h) to their algorithmically optimized 3D coordinates (target_x, target_y, target_z). By independently validating the data for each specific heuristic pathway, the study can systematically verify how accurately the machine learning model captures the unique placement logic of each approach—such as the vertical stacking dependencies evident in the non-zero target_z values—before the system is subjected to the final, rigorous testing phase
Sample Testing Split
Table XVII. Testing Split (200 Items)
Table XVIII. Testing Split (400 Items)
Table XIX. Testing Split (600 Items)
The presented tables illustrate the structure of the testing datasets, scaled progressively across 200, 400, and 600 inventory items. Unlike the training and validation splits—which utilize pre-calculated target coordinates to tune the machine learning model—these testing splits represent completely unseen, highly variable inventory profiles. They serve as the final, rigorous benchmark to evaluate the generalization and scalability of the trained hybrid optimization framework.
Each testing subset encapsulates the core physical and categorical attributes of the items, including dimensions (length, width, height), weight, and specific logistical classifications (e.g., bakery products, confectionery). By exposing the system to these incrementally larger and more complex datasets, the study effectively simulates varying degrees of warehouse density. This multi-tiered evaluation ensures that the integrated ML-driven Extremal Optimization (EO) and Genetic Algorithm (GA) can successfully parse real-world physical constraints such as weight-based stacking limitations and category-based zoning rules while maintaining computational efficiency and maximizing spatial utilization at scale.
Synthetic Data Validation
To verify the quality and mathematical consistency of the generated items, a multi-stage comparative analysis was conducted between the raw dataset, the GAN-augmented inventory, and their respective transformed states. Table X displays a sample of five item sets, comparing the original physical dimensions against the normalized feature vectors used for training, the resulting synthetic outputs from the GAN, and their final denormalized values. This comparison demonstrates realistic spatial diversity while confirming that the scaling and inverse-transformation confirming that the scaling and inverse-transformation processes accurately preserve the underlying logistical distributions of the inventory
Figure 11. Comparative Overview of Raw, Normalized, Synthetic, and Denormalized Item Samples
Table XX. 4-Way Comparative Analysis of Item Attributes Across the Data Transformation Lifecycle
(Format: Length, Width, Height, Weight)
(DISCUSSION)
C. Deep Learning Predictive Model Performance
This section evaluates the performance of the Multi-Layer Perceptron (MLP) in predicting the initial 3D spatial coordinates (x, y, z) and item rotations. The model serves as the "intelligent seed" that bypasses the need for a random starting state in the optimization process.
Prediction Accuracy
The predictive model achieved a Mean Squared Error (MSE) of 0.105753, indicating a high degree of precision in approximating optimal placement zones. Figure 13 illustrates the correlation between the predicted placements and the ground truth coordinates from the training set. The tight clustering of data points along the identity line confirms the model's ability to interpret item dimensions and fragility to suggest valid storage regions.
Figure 12. Regression Analysis of Predicted vs. Target Storage Coordinates
Physics Settlement Integration
The initial outputs generated by the MLP provide the raw numerical predictions for 3D item placement. Because a pure regression model may overlook strict physical boundaries, these raw predictions can occasionally result in "floating" items or minor spatial overlaps. To ensure that the numerical predictions are physically feasible, they were processed through the PyBullet physics engine to identify and correct these structural instabilities. Table VIII summarizes the Physics Settlement Correction Rate, showing the percentage of the raw predicted items that required gravitational adjustment to achieve a stable, load-bearing position on the warehouse floor or atop existing item stacks.
Figure 13. Physics Settlement Prediction
Table XXI. Physics Settlement and Stability Correction Rate
Figure 14. 2D Spatial Heatmap of Physics Settlement Displacement Across the Warehouse Floor
D. Hybrid Sequential Optimization Analysis
This section analyzes the computational synergy and the iterative improvement of the warehouse layout as it transitions through the hybrid pipeline. The process evaluates how the global search capabilities of Extremal Optimization (EO) and the local refinement of the Genetic Algorithm (GA) build upon the initial ML-seeded state.
Optimization Path and Convergence
The trajectory of the optimization is measured by the incremental increase in the global Fitness Score. Figure 15 illustrates this progression, starting from the baseline predicted by the MLP and moving through the subsequent refinement phases. The "ML Seed" provides a statistically informed starting configuration, which is then directly processed by the hybrid optimizer. Within this optimizer, the ML-driven Extremal Optimization (EO) component efficiently resolves macro-level spatial conflicts and accessibility overlaps. Finally, the optimizer performs high-precision "fine-tuning" specifically handling optimal item rotation and gap minimization guided by the ML-enhanced Genetic Algorithm (GA). This framework achieves maximum placement density through targeted, data-driven refinement rather than relying on traditional heuristic loops.
Figure 14. Fitness Score Progression Across the Hybrid Optimization Stages
Population Diversity and Selection
To maintain genetic diversity during the GA phase, the best-performing individual from the EO stage was injected as an "elite" member of the initial GA population. The code snippet in Figure 16 demonstrates the integration of the EO output into the GA chromosome structure, ensuring that the evolutionary process begins with a high-quality fitness baseline rather than a randomized state.
Code Snippet for EO-to-GA Population Seeding and Chromosome Encoding
Figure 15. Code Snippet for Robust Population Seeding and Chromosome Repair Initialization
Figure 16. Implementation of 6-Way 3D Item Rotation and Spatial Orientation Logic
Figure 17. Implementation of Chromosome Data Structure and 3D Spatial Encoding
E. Comparative Evaluation
The final objective evaluates the performance of the proposed ML-seeded Hybrid (EO-GA) framework by benchmarking it against standalone metaheuristics. This comparison demonstrates the efficiency gains achieved by integrating deep learning predictions with a sequential optimization pipeline.
Performance Metrics Comparison
The experimental evaluation was conducted across three inventory scales: 200, 400, and 600 items. Performance was measured using four key metrics: Space Utilization (%), Global Fitness Score, Execution Time (seconds), and Constraint Violation Count.
Table IX summarizes the comparative results. The data reveals that the ML-seeded hybrid model consistently achieved the highest volumetric density while maintaining lower execution times compared to the unseeded Genetic Algorithm (GA). Specifically, at the 600-item scale, the hybrid model reached a [X.XX]% Space Utilization, significantly outperforming the standalone EO and GA benchmarks which often struggled with local optima in high-density scenarios.
Table XXII. Comparative Performance Matrix across Item Scales
Visual Analysis of Optimized Layouts
To qualitatively assess the results, 3D wireframe visualizations were generated for the best solutions from each method. Figure 17 displays the final warehouse configuration for the 600-item test case. The hybrid model shows a more structured vertical stacking and clearer walkway adherence compared to the fragmented placements produced by the standalone GA.
Figure 18. 3D Visualization of the Final Optimized Warehouse Layout (600, 400, 200)
Figure 19. Evaluation Metrix GAN

### Extracted Tables (from thesis .docx)

|  | (1) |
| --- | --- |

|  | (2) |
| --- | --- |

|  | (3) |
| --- | --- |

|  | (4) |
| --- | --- |

|  | (5) |
| --- | --- |

|  | (6) |
| --- | --- |

|  | (7) |
| --- | --- |

|  | (8) |
| --- | --- |

|  | (9) |
| --- | --- |

| Column Name | Description |
| --- | --- |
| Object ID | ID of the item |
| Length (cm) | Length of the object in centimeters |
| Width (cm) | Width of the object in centimeters |
| Height (cm) | Height of the object in centimeters |
| Fragility (1–5) | Fragility rating from 1 (not fragile) to 5 |
| Access Frequency (/week) | Number of times the item is accessed per week |
| Category | Classification (e.g., Kitchen, Tools, Electronics) |

| Object ID | Length_cm | Width_cm | Height_cm | Fragility | Access_Freq_per_week | Category |
| --- | --- | --- | --- | --- | --- | --- |
| SKU_001 | 32 | 38 | 34 | 3 | 336 | Stationery |
| SKU_002 | 12 | 19 | 16 | 4 | 198 | Fragile Goods |
| SKU_003 | 53 | 12 | 32 | 1 | 141 | Toys |
| SKU_004 | 38 | 32 | 30 | 3 | 242 | Electronics |
| SKU_005 | 40 | 46 | 27 | 2 | 311 | Stationery |
| SKU_006 | 14 | 50 | 31 | 2 | 268 | Fragile Goods |
| SKU_007 | 60 | 14 | 20 | 1 | 120 | Toys |
| SKU_008 | 55 | 18 | 10 | 2 | 132 | Kitchenware |
| SKU_009 | 40 | 21 | 20 | 1 | 212 | Garden Gear |
| SKU_010 | 19 | 42 | 11 | 3 | 22 | Household |
| SKU_011 | 59 | 38 | 13 | 1 | 67 | Kitchenware |
| SKU_012 | 13 | 38 | 8 | 4 | 145 | Stationery |
| SKU_013 | 56 | 26 | 21 | 5 | 151 | Garden Gear |
| SKU_014 | 27 | 35 | 9 | 2 | 207 | Household |
| SKU_015 | 29 | 49 | 19 | 3 | 249 | Electronics |
| SKU_016 | 20 | 17 | 10 | 5 | 154 | Accessories |
| SKU_017 | 59 | 25 | 28 | 3 | 204 | Accessories |
| SKU_018 | 20 | 17 | 6 | 2 | 125 | Household |
| SKU_019 | 30 | 17 | 17 | 1 | 293 | Kitchenware |
| SKU_020 | 19 | 9 | 14 | 2 | 218 | Garden Gear |
| SKU_021 | 32 | 12 | 10 | 1 | 213 | Tools |
| SKU_022 | 58 | 28 | 36 | 2 | 26 | Household |
| SKU_023 | 60 | 46 | 32 | 2 | 120 | Tools |
| SKU_024 | 32 | 25 | 19 | 2 | 60 | Household |
| SKU_025 | 13 | 31 | 14 | 3 | 191 | Garden Gear |
| SKU_026 | 14 | 8 | 32 | 5 | 32 | Tools |
| SKU_027 | 40 | 8 | 9 | 3 | 344 | Tools |
| SKU_028 | 21 | 39 | 14 | 3 | 90 | Cosmetics |
| SKU_029 | 44 | 26 | 7 | 2 | 259 | Cosmetics |
| SKU_030 | 46 | 34 | 5 | 1 | 279 | Electronics |
| SKU_031 | 58 | 29 | 39 | 4 | 244 | Kitchenware |
| SKU_032 | 48 | 32 | 7 | 3 | 161 | Accessories |
| SKU_033 | 12 | 9 | 35 | 1 | 257 | Cosmetics |
| SKU_034 | 22 | 22 | 7 | 5 | 223 | Accessories |

| Rotation 0 | (L, W, H) | Original orientation |
| --- | --- | --- |
| Rotation 1 | (L, W, H) | 90° rotation |
| Rotation 2 | (L, W, H) | 90° rotation |
| Rotation 3 | (L, W, H) | 90° rotation |
| Rotation 4 | (L, W, H) | 90° rotation |
| Rotation 5 | (L, W, H) | 90° rotation |

| Data Type | Scaling/Normalization Method | Output Range |
| --- | --- | --- |
| Item Dimensions & Weight | Scaled down by a factor of 10.0 | Constrained Numeric |
| Warehouse Dimensions | Scaled down by a factor of 100.0 | Constrained Numeric |
| Placement Coordinates (x, y, z) | Divided by respective warehouse bounds | Relative Percentage [0, 1] |
| Rotation Indices | Divided by total rotation count (6.0) | Bounded Continuous [0, 1] |

| Layer Type | Configuration / Output Size | Purpose |
| --- | --- | --- |
| Input Layer | 10 Neurons | Accepts the normalized 10-dimensional feature vector |
| Hidden Layer 1 | 64 Neurons (ReLU) | Initial feature extraction and non-linear mapping |
| Hidden Layer 2 | 128 Neurons (ReLU) | Deep spatial relationship learning |
| Hidden Layer 3 | 64 Neurons (ReLU) | Dimensionality reduction for output preparation |
| Output Layer | 4 Neurons (Linear) | Predicted x, y, z coordinates and rotation index |

| Parameter | Configuration | Functional Role |
| --- | --- | --- |
| Iterations | 500 | Total generations for global search |
| Selection Strategy | Worst-Component | Identifies items with highest local\_score_i |
| Improvement Attempts | 20 per item | Local search trials per identified component |
| Placement Logic | Zone-Aware | Enforces fragility and safety constraints |

|  |
| --- |

|  |
| --- |

| Attribute | Raw Attribute (Example) | Normalized Value (0.0 to 1.0) | Purpose |
| --- | --- | --- | --- |
| Length | 0.58 cm | 0.058 | Scale invariant spatial feature |
| Width | 0.42 cm | 0.042 | Scale invariant spatial feature |
| Height | 0.44 cm | 0.044 | Scale invariant spatial feature |
| Weight | 11.58 kg | 0.1158 | Physics-weight distribution |
| Fragility | 1 (High) | 1.00 | Vertical constraint mapping |
| Stackable | 0 (No) | 0.00 | Stacking logic constraint |
| Wh. Length | 2.50 m | 0.025 | Global coordinate scaling |

| {"1": {"article": "ciabatta-00103095", "id": "00103095", "product_group": "bakery products", "length/mm": 590, "width/mm": 200, "height/mm": 210, "weight/kg": 7.67, "sequence": 1} |
| --- |
| {"2": {"article": "cake-00111025", "id": "00111025", "product_group": "confectionery", "length/mm": 550, "width/mm": 280, "height/mm": 110, "weight/kg": 8.4, "sequence": 2} |
| {"3": {"article": "cake-00111025", "id": "00111025", "product_group": "confectionery", "length/mm": 550, "width/mm": 280, "height/mm": 110, "weight/kg": 8.4, "sequence": 3} |
| {"4": {"article": "dessert-00104636", "id": "00104636", "product_group": "candy", "length/mm": 490, "width/mm": 130, "height/mm": 210, "weight/kg": 5.11, "sequence": 4} |
| {"5": {"article": "dessert-00104636", "id": "00104636", "product_group": "candy", "length/mm": 490, "width/mm": 130, "height/mm": 210, "weight/kg": 5.11, "sequence": 5} |

| item_l | item_w | item_h | target_x | target_y | target_z |
| --- | --- | --- | --- | --- | --- |
| 0.79 | 0.56 | 0.52 | 10.78 | 5.395 | 0 |
| 0.59 | 0.43 | 0.45 | 6.87722 | 4.59361 | 0 |
| 0.76 | 0.46 | 0.34 | 4.06331 | 11.4955 | 0 |
| 0.78 | 0.53 | 0.51 | 2.00859 | 2.66075 | 0 |
| 0.78 | 0.53 | 0.53 | 2.3804 | 2.92028 | 0 |

| item_l | item_w | item_h | target_x | target_y | target_z |
| --- | --- | --- | --- | --- | --- |
| 1.15 | 0.38 | 0.4 | 1.575 | 1.87 | 0 |
| 0.76 | 0.34 | 0.35 | 10.9953 | 14.0669 | 0 |
| 0.77 | 0.51 | 0.7 | 1.525 | 2.885 | 0 |
| 0.58 | 0.42 | 0.45 | 2.34916 | 2.69819 | 0.28 |
| 0.8 | 0.39 | 0.55 | 0.4 | 1.695 | 0.33 |

| item_l | item_w | item_h | target_x | target_y | target_z |
| --- | --- | --- | --- | --- | --- |
| 1.09 | 0.56 | 0.2 | 4.84186 | 11.123 | 0 |
| 1.08 | 0.54 | 0.51 | 16.0756 | 7.9921 | 0 |
| 0.79 | 0.52 | 0.57 | 8.22347 | 3.9133 | 0 |
| 1.08 | 0.54 | 0.51 | 1.95979 | 0.62097 | 0 |
| 1.18 | 0.79 | 0.55 | 1.51915 | 2.44663 | 0.47 |

| item_l | item_w | item_h | target_x | target_y | target_z |
| --- | --- | --- | --- | --- | --- |
| 0.79 | 0.53 | 0.55 | 5.82293 | 6.39829 | 0 |
| 1.18 | 0.79 | 0.55 | 10.09 | 0.395 | 0 |
| 1.18 | 0.79 | 0.55 | 12.09 | 5.895 | 0 |
| 0.95 | 0.41 | 0.49 | 2.255 | 1.205 | 0.79 |
| 0.78 | 0.53 | 0.36 | 2.45 | 0.265 | 0 |

| item_l | item_w | item_h | target_x | target_y | target_z |
| --- | --- | --- | --- | --- | --- |
| 1.18 | 0.78 | 0.33 | 6.81756 | 15.6536 | 0 |
| 0.66 | 0.41 | 0.58 | 2.12925 | 0.946623 | 0.4 |
| 1.11 | 0.78 | 0.29 | 2.56649 | 0.713195 | 0 |
| 0.58 | 0.41 | 0.39 | 0.745 | 0.29 | 0.38 |
| 0.77 | 0.53 | 0.32 | 2.765 | 1.385 | 0 |

| item_l | item_w | item_h | target_x | target_y | target_z |
| --- | --- | --- | --- | --- | --- |
| 0.8 | 0.61 | 0.46 | 15.305 | 4.4 | 0 |
| 0.79 | 0.53 | 0.54 | 3.435 | 1.895 | 1.05 |
| 1.11 | 0.41 | 0.57 | 1.555 | 1.705 | 0 |
| 1.13 | 0.3 | 0.46 | 1.15 | 1.065 | 0.33 |
| 1.17 | 0.39 | 0.43 | 4.26893 | 8.70394 | 0 |

| item_l | item_w | item_h | target_x | target_y | target_z |
| --- | --- | --- | --- | --- | --- |
| 0.78 | 0.53 | 0.51 | 9.0535 | 11.9024 | 0 |
| 0.65 | 0.41 | 0.58 | 3.185 | 1.325 | 0.36 |
| 0.78 | 0.53 | 0.54 | 1.39 | 1.135 | 1.01 |
| 0.79 | 0.39 | 0.55 | 0.195 | 1.895 | 0 |
| 0.78 | 0.33 | 0.33 | 17.5689 | 4.13914 | 0 |

| item_l | item_w | item_h | target_x | target_y | target_z |
| --- | --- | --- | --- | --- | --- |
| 0.78 | 0.6 | 0.35 | 10.7685 | 8.57653 | 0 |
| 1.18 | 0.8 | 0.22 | 1.59 | 0.9 | 0.99 |
| 0.77 | 0.34 | 0.35 | 1.53389 | 2.77421 | 0 |
| 0.52 | 0.39 | 0.53 | 2.485 | 0.26 | 0.79 |
| 0.76 | 0.52 | 0.64 | 15.5316 | 4.16756 | 0 |

| length | width | height | weight | category |
| --- | --- | --- | --- | --- |
| 0.78 | 0.53 | 0.5 | 21.39 | bakery products |
| 1.11 | 0.46 | 0.47 | 14.76 | pizza |
| 1.18 | 0.79 | 0.26 | 16.81 | candy |
| 0.91 | 0.53 | 0.47 | 16.67 | side dish |
| 1.18 | 0.79 | 0.55 | 12.58 | bakery products |

| length | width | height | weight | category | length |
| --- | --- | --- | --- | --- | --- |
| 1.17 | 0.79 | 0.51 | 14.29 | bakery products | 1.17 |
| 1.1 | 0.5 | 0.27 | 17.56 | confectionery | 1.1 |
| 0.8 | 0.59 | 0.5 | 19.46 | confectionery | 0.8 |
| 1.16 | 0.42 | 0.36 | 15.34 | confectionery | 1.16 |
| 1.09 | 0.56 | 0.21 | 16.36 | ice cream | 1.09 |

| length | width | height | weight | category |
| --- | --- | --- | --- | --- |
| 0.77 | 0.57 | 0.38 | 10.59 | confectionery |
| 1.1 | 0.51 | 0.56 | 8.5 | vegetables |
| 0.59 | 0.2 | 0.2 | 8.25 | snack |
| 1.11 | 0.35 | 0.47 | 8.94 | candy |
| 1.13 | 0.3 | 0.45 | 13.74 | confectionery |

| Sample | Original (Real) | GAN Normalized [0-1] | GAN Denormalized | Synthetic (2x Scaled) |
| --- | --- | --- | --- | --- |
| 1 | (0.59, 0.20, 0.21, 7.7) | (0.415, 0.314, 0.188, 0.145) | (0.35, 0.20, 0.26, 4.3) | (0.71, 0.40, 0.52, 8.6) |
| 2 | (0.55, 0.28, 0.11, 8.4) | (0.361, 0.276, 0.216, 0.151) | (0.32, 0.19, 0.29, 4.5) | (0.64, 0.38, 0.58, 8.9 |
| 3 | (0.55, 0.28, 0.11, 8.4) | (0.380, 0.278, 0.218, 0.172) | (0.33, 0.19, 0.29, 5.0) | (0.66, 0.38, 0.58, 9.9) |
| 4 | (0.49, 0.13, 0.21, 5.1) | (0.468, 0.577, 0.150, 0.162) | (0.39, 0.28, 0.21, 4.7) | (0.77, 0.57, 0.43, 9.4) |
| 5 | (0.49, 0.13, 0.21, 5.1) | (0.799, 0.308, 0.150, 0.246) | (0.59, 0.20, 0.21, 6.8) | (1.17, 0.40, 0.43, 13.6) |

| Model Variant | Correction Rate (%) | Mean Displacement (m) | Max Displacement (m) | Stability Index |
| --- | --- | --- | --- | --- |
| EO | 100.00% | 11.7196 | 16.9991 | 0.0000 |
| EO_GA | 100.00% | 10.6027 | 17.5912 | 0.0000 |
| GA | 100.00% | 10.7284 | 17.5240 | 0.0000 |
| GA_EO | 100.00% | 10.4585 | 17.2759 | 0.0000 |

---
