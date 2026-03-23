# Related Literature for Training-Bin-Packing System

This document maps academic literature and theoretical concepts to the specific components of our 3D bin packing system.

## 1. 3D Bin Packing Fitness Function (Physics & Stability)
**Related Code Component:** `optimizer.py` (specifically `fitness_function_numpy`, `repair_solution_compact`, and stability checks)

Our system uses a sophisticated fitness function and repair heuristics that account for:
- Space Utilization (Volume efficiency)
- Accessibility (Distance to doors/handlers)
- Stability/Geometric Support (Ensuring items are supported by the floor or other items)
- Stackability (Respecting item fragility constraints)
- Overlap Penalty (Ensuring strictly non-overlapping placements)

The following literature discusses multiobjective fitness functions, physical constraints, and heuristic guidance in 3D bin packing:

*   **Stability and Load-Bearing in 3D Bin Packing:** Ensuring stability is crucial to prevent safety hazards. Academic approaches range from simple heuristics (heavier items at bottom) to complex physical modeling like Load-Bearable Convex Polygon (LBCP) methods, which align with our `is_stable` geometric support constraints.
    *   [Source: Physical Modeling in Bin Packing Optimization](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFKUjW72BSaXb-A2JTfJWiU9tEmkhqyZlKNCgQQj7H0DSf9Cj4o7uTPkVzAnkyGRTj_k04LZ_2p6384-cLcgM8YoP95XkKz_Boo0z6YZufxuGy-8E44Bf6iTIBi_y4=)
    *   [Source: Real-World Constraints and Spatial Fragmentation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF00QVbKpRMwC4w4xINr4GzfZ_5pqNUXGrXXU-VUDWD3SMzT3IZKHOdIrbBCk5DgHO1N0iDSMwlzbRzIgA_MEqn1gE9uhJT2slSuit9fMYayKp3q4p3kz7aQ8LfQcl5eVJId4shQQGndc4UvxBa)
*   **Space Utilization Optimization:** Literature emphasizes minimizing the number of bins and maximizing packing density, handling restrictions on orientation.
    *   [Source: Space Utilization & Orientation Algorithms](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGShpCTBR3RfnNW21UVZh86S0qAyEe1C1NNtnFBj0JjFVQ0Gh99OJwfytBe4z-aUdCvocaF9ziTa1D0MfUzI_ZdFPsHHETg2Pgnfph505QGVPdQXwscKlG157hvXCn-FJAyl5_1_2C2awMkgBLV)
*   **Hybrid Genetic Algorithms and Heuristics:** Our system utilizes both a Genetic Algorithm to evolve item placements and deterministic heuristics (like gravity-based Z-axis dropping) to repair solutions. Hybrid models in literature frequently explore combining heuristic initialization/placement with GA refinement.
    *   [Source: Hybrid Genetic Algorithms for Logistics and Optimization](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF_EUH-IjMPQcRNlP6XyoycqOPlIwfXwjAJoKZhhYZVdsbTNq2hNUF6wfIfye4Ucf_ZvDV4MX98ye0rHfIlWuVPM_bH0hyu90zxb3zRV-YK4Y2Rq2FJKlREKIAY2ZzKZKBuwIjmFe6476hqw1yVANKCgTOOLsU=)
    *   [Source: Heuristic-Guided 3D Bin Packing and Extreme Points](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHTw5EqRPubKSnk-MJFLxKqQ0DKa-QqEIfuRxIA69QvCyjbBW6cDyo5YxclMABg6RIeACab-11-TPYLdSLnWHVBteBklRywVWCKaHU7wocAWOzooHZdLjWne7kWTlmRJY9p50ShuRkgFgsl0hzOxbAO47yLvD8=)


## 2. Generative and Neural Combinatorial Optimization
**Related Code Component:** `gan/` directory (`model.py`, `train.py`)

Our system utilizes a Generative Adversarial Network (GAN) to generate packing combinations/training data. The generator creates feasible packing assignments from latent vectors.

The following literature explores the integration of GANs, Deep Reinforcement Learning (DRL), and generative inference for solving combinatorial problems like the 3D Bin Packing Problem:

*   **GAN-based Genetic Algorithms for 3D Bin Packing:** Research shows that integrating a GAN generator within a genetic algorithm framework significantly improves the exploration of the solution space. The GAN generates diverse, high-quality packing solutions that help traditional optimization algorithms avoid local optima.
    *   [Source: GANs in 3D Bin Packing (MPU)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFfLRaRRf15V_4GqrwV5zI2CZcL47W0FJK12B_VwY6T6ufGQIASGiJoUlR1Jqkhplj9zmA45ViiqI24NUK8ey8_TBc7vuf1L0qPVurQBoA0oScKBSvT4XxtbBCQXqR9Cj75ltUP8elrcU8GjwewAAaOJaWptvCplDkyWYlq3_Py1ozJMjDnwF6fMhDUA4RwDcM866R6FgEGQh170LUAezRaZQHH33taQJz-)
    *   [Source: Generative Models for Combinatorial Optimization](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGJ0AB2a6kvD6_2ArRmHsCm8DUKQKRdt7swF6YaoGkzasQeew01sh7bRY38OKRJAaDkk407NKgYNrlFC-LYGRePVQnqBqpgjeIyMX1YfL_h8VCTkY-6D25EmwmCYwR5v7B4OgxVpA5zAdEHCUTI)
*   **Combinatorial Optimization Inference:** Accelerating "inference" in machine learning models directed toward combinatorial optimization reduces computation time for generating massive action spaces.
    *   [Source: Inference in Generative AI for Optimization](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGk3hm6mzDZumbxfGu4MalikDKqKXxv-cHGIhuK30VVDKNDsjM0Lt-tIy8oF6yVeZ135SmyPmAyG6j5xPx4n91RpgUeQzpUdtGsawGM0tV4XN9lo5cYbbQTMF6vHqDcY4jUr23K8rX16m6n10jPbcUpamhzONchIGm9_3-CKhglHz2eUGmlYeaMMeqSxCf-B2dCMCG5N0hPTFiLWfdx4gAiKFYubbVB8atMVbDiUoZ5belW)
*   **Deep Reinforcement Learning & Neural Combinatorial Optimization (NCO):** Modern machine learning approaches formulate the 3D-BPP as a Constrained Markov Decision Process (CMDP). Like our GAN approach mapping latent vectors to outcomes, DRL agents learn to pack by utilizing Multimodal Encoders and Heuristic-Guided frameworks to respect real-world spatial and stability constraints.
    *   [Source: Deep Reinforcement Learning for 3D Bin Packing Applications](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGJ3DoydHi6P6AYVbf8JSBYgFoTab2tOl5Eb-BkUmaPUlCFGW2Ehi6cMW4xJ5Xjgz8csxx497TlNAyY6FL1zY4OYkk_jaC0Y1L1s0PV_IwJLBL2oKTnDQEeUEkIpUXsr-CotMVFwYgukP_cwxlEUwUo)
    *   [Source: Neural Combinatorial Optimization Methods for BPP](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHQiNsfjJOWdfSS4DxBJkTmfTDKlnfP6Rc10dwRQoUd2l8VTDG8kiESDfzv-WSCIEvy-4Q3mevAWTRqk3-QKmeZA0gv_4ZGoUvyUK3MJ2ki0mQAdWOJ-evrHSDBfOb_Xvs-5GKoUlBV1E-z8OWR2h3Zfg9Uq--EgbI=)


## 3. Alternative Optimization Baselines

While our system focuses on GANs and Genetic Algorithms, it is very common in literature to benchmark against or combine with these alternative optimization paradigms:

### Exact Methods & Mixed Integer Programming (MIP)
Because 3D Bin Packing is strongly NP-hard, mathematical exact methods are used as a ground-truth baseline but generally only scale to small numbers of items.
*   **MIP and Branch-and-Bound:** MIP models determine absolute optimal space utilization, but the computational cost explodes as items scale up. Researchers use tools like Gurobi to solve these constraint equations (which mirror our numpy-based overlap and stability penalties).
    *   [Source: Mixed Integer Programming for 3D Bin Packing](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHA8C0yPw_5WfjvJ7SSX_iK09vnLc1YOBGqgqfyFH_cp2uJ7yGBTVrn5qln7r39DqvD-ITPDrd-bzrTnfdFfIgGJfBg2YNzhZ9uhjmKjFITsIGFJbjJ0vuHk2vr7RWJNv0lQwMx93sQtS-sOwaopvf1wqreY4RYhZkhAKHXGwM0_d7wMI4=)
    *   [Source: Exact Methods Overview (Cutting planes, dynamic programming)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGHZDAPVeKtueS-VusXkVYACMjg-_3X-zRt22I61-aJ19yC_aEv7mLC6yQww-nKr6M-a9CWleaAc3khyjHxKsOrFXhhEaVq2NWeMBcXn0BdPdhckui7Nuje-DSIEmWlreL1pbKnP_4RoH7WMJv8KFGPcVndtq2SoWaQ785xjRQy0bAFmArFEvlPGgoIkQ==)

### Advanced Metaheuristics: Particle Swarm & Simulated Annealing
Like our genetic algorithm, Particle Swarm Optimization (PSO) and Simulated Annealing (SA) are metaheuristics popular for tackling 3D-BPP nonlinearities.
*   **Simulated Annealing (SA):** Deals well with the discontinuous objective spaces typical of stacking boxes by occasionally accepting inferior configurations to escape local optima (a problem GA also tries to solve with mutation).
    *   [Source: simulated annealing in Shipping Container Packing](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG3WyEFP9PQo8i9WTsPhOwcIwE6CZuJXSP6cPpNVbgXo2ylDVKayLMZ8cuhboL5lx-puvdMs2qgTMdhMtxn6TN-mupDHBb4NqC01uq6q5QFMQzFAGc3pvW6fXIMTUr43ibBGNpZIkDj5NEQKdWAsp1wlR8gUW97iPTUalApl37CPK437umM0D9AzKcgaWKo_SiWiWzARNo6PgsI-HD55fxpaMAme382z0gcAtDPUhU3aHBfUA==)
*   **Particle Swarm Optimization (PSO):** Models the solution space continuously. Hybrid Multiobjective PSO algorithms (HMOPSO) map well to our multi-criteria fitness function (evaluating space vs accessibility vs stability).
    *   [Source: PSO-based Joint Search Heuristics](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFoz5YlWvD3A_9sWDbQqEMYsxImCa_rbQJQjsQ2baPiEoUlgsJ_nY9suqHrI0NDzRlc3RsEp8mLe6MMRqB0hDOoslMv4OT3t46LI1JFW6SaKLbuyhY8z0lBorYqIv68fmg=)

## 4. Benchmark Datasets for 3D Bin Packing
**Related Code Component:** `datasets/` directory (specifically `bed-bpp_v1.json` and `convert_dataset.py`)

A crucial part of bin packing research is evaluating algorithms against standardized datasets. Our system uses a parser to convert instances (like the BED-BPP format).

The following literature discusses standard benchmark datasets used to train and evaluate 3D Bin Packing algorithms:

*   **BED-BPP (Benchmarking Dataset for Robotic Bin Packing Problems):** The dataset format explicitly targeted in your `convert_dataset.py`. This dataset represents real-world item sizes and targets from grocery companies, explicitly designed for both offline and online learning environments.
    *   [Source: BED-BPP Official Overview](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEGUS3VVDukTQRAlTtp_cpbdd3eR-5QlhI6jveBmOIkFoiyOCGd45S6_jVe0Igw683fJ3DgaFGt27g-_ZVmFFTBOwBGx201wZMOvAWxQMPfruLGAulcLBfQx10hLMjuWAU9u5OFkTLiVVpI0mzXp63cq88QMWjhvUvjbHmFrwE0rqUlIpqwEEg=)
    *   [Source: Benchmarking Robotic Bin Packing](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGoaOSxK614DzuDj5VGfnR1sus6odqHxdczGAqhKMuDMJrCNl_NwYjG8B047bQm5Cpl6qAEp9-RpYOW7Zt9nyKyLtWKE_cjfw1eD0U8jdN1VnCFXNN4CObCVufTPwvRgMr14a7e)
*   **Bischoff and Ratcliff / OR-Library Instances:** The most famous classic datasets for single container loading (often labeled BR0-BR15 or `thpack`). They are widely used as the baseline comparison for metaheuristic performance in maximizing volume utilization.
    *   [Source: Bischoff and Ratcliff Baseline Overview](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEp3cLCTxSsD0qYEhUXsOHfkQ9jv_dlvSXGExQtQkntt1_PxzNFalTi2qBRmdg7l11-gfZOfICtW-MW1z4IVThlm0JNJkiwtDSMbzfBa9MzvdI-RQrbB3GfBuHOkD2olx0YwXlT3ki0UXyBXGHcBtMYMMWp6EflmWDxZ9AiJZ56PGCVgU6UAjqGBjitkX0PG7a7_83lhrSBY2yK2lglrxt7cHoadJViYm-UlzWtPdo8_YSPWqBFlVCXylx4plmphwH02bGMXoDnBgNgJK5YCj5Yo6eW-zgdZPzXXLs=)
    *   [Source: OR-Library Datasets](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHNSpHFZ97fCFxqO_Gct1r94WhjewAifuhRKu4eVoMDZPave8pqYrOSO0vsSiCmEHF78Mxl-ud8F71lf9bAbgFT9NYof6PNOOYchb1pywU9pG2AStfS5NTMKmC7Z-RZrvKOIc3U4ysCNTtbVmweYtTvU_a65OFytuPELepKPWgQmsdvSIp1Fqigp4aiFLwRWg3LHtSmwnn5pk6h70_QnUPYevHeqqT4x0EkjtVPUfj5BkZLLRtYuqb0pSrhRI1LHkdppp6bFutNgjKUalex)

## 5. Multi-Objective Constraints (Accessibility & Grouping)
**Related Code Component:** `optimizer.py` (specifically `access_scores`, `freqs`, and `grouping` calculations)

Your system grades bins not just on volume, but on accessibility (frequently accessed items near the door) and grouping (keeping same categories together). This elevates the problem from a standard 3D-BPP to a Multi-Objective 3D-BPP.

*   **Accessibility and Picking Efficiency:** Packing algorithms must account for the operational cost of picking. Items needed sooner or more frequently should not blocked by others.
    *   [Source: Picking Constraints and Access Efficiency](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEkLKglL_7rrONRNrq8ziJinkKnt0txj2fEAIE9AHKb0P3mqrotfzgPHcI4ER2gt679nS6nPrakXq036vo2iWrjXMhfY7IFfCOJmOnC03pZ5IMxz8SGqaAfTmBfEK88hwD14omFbAfd3LFnCHSJnbWH_AlyAQ==)
    *   [Source: Robotic Bin Packing Operational Constraints](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH3otEYgy8hIDmi9E-N4DaNfGPsje9ymSRm3H8i_ZPS0jYxBlCQ9PPQzqzUICwrIRSYbeLG-bjvTuPf_p6EyCPEIe2QvNGEEAdNAulC1sQr5XEqaKt1UqLIfDV__15u)
*   **Grouping and Product Family Constraints:** Real-world logistics often require items destined for the same customer (or the same product family/category) to be packed together in the same bin or layer.
    *   [Source: Product Family Grouping Constraints](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGsKqhn9ZQ0gAkvvfBuPwNR11OJV_FGgnM0y7AE9kTzqTeb5uYKNKaptfIF3LEZJCHV5Z8dyjLcesVYR0hOur9xhVpeEey3PlwXkUOndWJWsntpyDVha0OcaiJBpfNmZhlS7dTL4YXu05a5SSO4jDYeH7fFIzhpXu-75oUCm44e_KA9SWshAhc0nwmH33kKM-yfJdppcWXb4JaX8as0L_EANgA4YfgRYtHsuHIhwlqfwL2vcwUdZ207Vg==)

## 6. Geometric Packing Heuristics
**Related Code Component:** `optimizer.py` (specifically `repair_solution_compact` iterating by layers/Z-heights)

While machine learning dictates the rough layout, greedy heuristics are often used to ensure the final output fits perfectly.
*   **Layer-Building and Wall-Building:** These are the most common deterministic heuristics to generate packed bins. Layer-building forms horizontal planes (grouping items by height), while Wall-building groups items by establishing vertical/horizontal planes from the far walls. These act as fast fallback mechanisms (similar to your repair logic) when pure mathematical optimization falls short.
    *   [Source: Layer-Building Packing Patterns](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEtTXEhr0z7DMsLf_qySFq4U4ta-qVDFewDwxmLI5Tnv1KIs_quCDdok0PqFun-PJyFctHYMI-fiMk3PfSDlNvgqmW25WxWzJKhCBn0ccrXNxWnh9Uiw2s8-vxKXpNnFoZ81uRJvBcgUbRiEypnyDQVrqRyiSXalPj7mhp_wVMcwy0_jchoMrPIot_xvWJMg4qh-bjrurM0J4buBsqoWRXfnxms8ZAXRM8SR6E0xzqBGczaBg9IfXEzvGwXu1NM6xeAswogbYiBTDz60-u7r47yZFSbIwFI_ko-fCWuynQ5k650xy-jdpE=)
    *   [Source: Wall-Building Heuristics as Fallback](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGxGtCaKbwROaSAGhf0GuLVH7v0ykFFBwft2SvX98B0-PcIdEWS52Is7KZt5-fx9gQ_gwtzUn13kjD_Drvj7pYIf1FEly0SUnrNq1SIxVbeN20jF8H-Ph9xfLdK1tKNZdenG8w5l7wy-gS5HyX_3wkA_g29uh4=)

## 7. Physics Simulation for Packing Validation
**Related Code Component:** `optimizer_physics.py` (PyBullet rigid-body settling)

Your system uses **PyBullet** to run a headless physics simulation that settles items under gravity, resolving overlaps and ensuring physical feasibility. This is a growing research area.

*   **Physics-Enabled Bin Packing Environments:** Researchers build PyBullet environments for Online 3D-BPP, simulating realistic item settling (toppling, gaps, orientation changes) to train RL agents and validate heuristic solutions.
    *   [Source: PyBullet Environment for Online 3D-BPP](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH0RoZbni-joWx8uBAHltbO0HKMfIR-F0cR_OkFL_3rsBTkkrdX7QSpw2bdoZJFrUstdTDdKZmmvh4AmXnGAVeoGGGKfs64g8ERQOTZW2WRsAHffCqhg-WTGJczZpoKfj5uCrVUmA4AVtnxjdqzdD8PCI-JrgBUAh-b6gOwlYCRQRAIErv8ndZa7t6aM6fbNHbZHfeuiKkaOwPJNW3ssORJdtQgBODFl7xYW6TWB8d7k8iQO55TvvvX0ogy6vCFcWeu70zAr8fbItsvtFOtldiWJOGqxeLdkmetho6nNP9f)
*   **Rigid-Body Stability and Settling:** The simulation parameters (friction, CCD, solver iterations) directly impact whether stacked items remain stable. Literature discusses Stewart-Trinkle methods and deactivation/sleep for computational efficiency.
    *   [Source: Box Stacking Stability in PyBullet](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH6ndBJGMG9y06ri-rTSiZobL8-W4n3xcarj9K4_1jl8bUAMFW6IR6S3syiZgxUPnObT3JVANRbwEC1x-BC-IuEeU0lxp4Ifw-a_GZbfSs7nxDrlbrH588WTFVcAOVMquyP8K3EEhyKdtQDdUTaB8XVEA==)
    *   [Source: Stable Stacking Without Sleep States](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHHRCY1F2fQlPL8B7m8p2FQocm79jyeV8gFGyJomiYcW8IQ4W7Zo32tdwAnu5tDidfoWg0vesRBGZgrGUVj2ANVaL-xDfsYouEX8LQKNqAHU3AUuDLnoumcZGRcJm8-jYB1UJaz8aHaH1h87mDd1YZu8Q==)

## 8. Supervised Learning for Placement Prediction
**Related Code Component:** `ml_utils.py` (`PackingModel` neural network and `MLOptimizer`)

Your system trains a feedforward neural network on optimizer-generated solutions (from `generate_training_data.py`) to directly predict item (x, y, z, rotation) placement from item features + warehouse dimensions.

*   **Neural Network Item Placement Prediction:** Research explores using encoder-decoder neural networks and Markov-chain-based sequence predictors to learn packing strategies from human demonstrations or algorithm outputs. Some approaches decouple orientation from position prediction.
    *   [Source: Learning Packing Sequences from Demonstrations](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFkJJ6EQ14TkTDVMm9QhbQiiFWxA5zZvj0sYasLfPHXFiXlxrca-r3lGRLHNZqwdr5vFDGvvMyaoNMS5TSypgh9EFPe6U_U0xZjfqAlCQixj0nEncec62t7jpjDoFFzhbD0fzlErhkZPD8gFhSI0-xba80mB7e7wOmQMLY3siGxV6YkPWLrhvFXB6StroKQV63e)

## 9. GAN-Based Synthetic Data Augmentation
**Related Code Component:** `gan/generate.py` (synthetic item generation using trained Generator)

Your GAN generates synthetic items (dimensions, weights) that match real-world distributions from the BED-BPP dataset, then applies heuristic rules for fragility, stackability, and rotation constraints.

*   **GANs for Logistics Data Augmentation:** GANs address data scarcity in logistics by generating realistic synthetic datasets for demand forecasting, warehouse layout optimization, and inventory planning. The adversarial training produces distributions indistinguishable from real operational data.
    *   [Source: GANs for Supply Chain and Logistics](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEWg-ZqvWCEv06YyV_BDYdcK8mT3QzO7COQEe1ba7axjrCPCj72-g59J9zZyz6ZDVeVVS-vaLZ3OoDhZKAS8GTa73UBc-irGFQGwvT-Zhx59214Kc9NTRu5dpTEVJUJj3nuvsvt1mCzwH_CB-UUIA0=)
    *   [Source: Synthetic Data Generation with GANs](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG4iVXn1hRmFjKd_jmpJOpE7N5nroQ9HVi5v66xUO18tky_-9VUokQI8gMgjCnwL8LIs_1uYFF1UC-3xnk_JBPu8ykeCwInRmJ5sleFeI3O0iYOLDyq5MDSFnctXyVkSF2zPg5GwMQXS7EJA0_BXbyJW_4dE05OOCVGqvTHr7EosWR57j4tj0UX6x06EEjwZ5z5N_jBY3oQnQ2pvt4_Pz9xwgkAIUTKKfiY7GkuIxQ80GpWXvTeutEEIOMsmHVIz7w=)
*   **GANs in Estimation of Distribution Algorithms (EDAs):** GANs can replace traditional probability models in EDAs for combinatorial optimization, generating candidate solutions that capture complex variable dependencies.
    *   [Source: GAN-EDA for Combinatorial Optimization](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHZgC21zvTs8HAz8CH6QGFZk9npbOyMuolytrEXuv9-N_9QQbffIm-seUd62DlrmM_l24H2XjI9-PMPpGeV7xQx2mT5kk0UB546x3f6Me07SsAChJX8dIbn42jJF0xJn5tBMS7gW8mulL3aaPpPd1GdjfqkWqhCp1BoxjGQVedUExEBdslFDw==)

## 10. Extremal Optimization (EO)
**Related Code Component:** `optimizer.py` (`ExtremalOptimization` class and `HybridOptimizer`)

Your system implements Extremal Optimization as an alternative to the Genetic Algorithm and also in a hybrid EO+GA mode. EO is a local-search heuristic inspired by the Bak-Sneppen model of self-organized criticality.

*   **Extremal Optimization for Bin Packing:** EO evolves a single solution by iteratively replacing its worst component. Hybrid EO (HEO) approaches combine EO with improved local search to escape local optima and traverse infeasible solution space, achieving competitive results on large BPP instances.
    *   [Source: EO Algorithm Overview (Wikipedia)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGRHuVAujQb6Yy6xcsYfXLSyQWBlr0SI9dsaGTegZPCXDsuCfdqVrWe1h4vrS_PBkjyzkpaJhSIQst0V3HkIOQfxKVIfTxHB7Esq1LHpEuo4N3HXXJNys-jV9467QKwyCqivfMeuA9o4YyiPlhS)
    *   [Source: Hybrid Extremal Optimization for BPP](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHWJ3IZcxfdOjTc0JZnQuTjjjwAmSfFSQaOnPLUg8c7Y6ZOva2bmidJVKSatXfhSsiUj8Iak6_rCt3MglHYnHTIpNT3CIT2yl6GcFxGc9tBjct7fCAFWi73RRvkMnl_Tazq1o1F20k1LXc7xxk2JEb4vk0E2V1xh4B-9PRM4X5S7amo9bYnc3_X6Q6hGSKp2k9owyXq_QuErNuQdQ11ar_pv51BT1G6fJCUBuy-pBWz)
