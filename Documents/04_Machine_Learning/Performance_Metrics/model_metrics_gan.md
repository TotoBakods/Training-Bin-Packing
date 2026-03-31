# GAN Performance & Generation Report

> Auto-generated on **2026-03-31 18:06**

---

## 1. GAN Training Foundation
The generative foundation consists of a Generator/Discriminator pair trained for **500 epochs** to synthesize realistic warehouse SKUs.

### Training Metadata
- **Epochs**: 500
- **Batch Size**: 64
- **Hardware**: NVIDIA GeForce RTX 3060

### Stability & Convergence
![GAN Loss Curves](metrics_visuals/gan_loss_curves.png)

### 1.1 Methodology: Min-Max Scaling
To ensure stable training, all physical dimensions are normalized using **Min-Max Scaling** to a strict **[0, 1] range**. This matches the Generator's `Sigmoid` output layer and prevents any single feature (like weight) from dominating the loss function due to its different numerical scale.

| Phase | Initial Loss | Final Loss | Parity (D/G) |
|-------|--------------|------------|--------------|
| Discriminator | 0.6837 | 0.6782 | 0.0218 |
| Generator | 0.7336 | 0.7386 | 0.0386 |

## 2. Synthetic Dataset Generation Logs
The following datasets were generated for final inference benchmarking:

| Dataset | Item Count | Avg Length | Avg Width | Avg Height | % Stackable |
|---------|------------|------------|-----------|------------|-------------|
| `200_items.csv` | 200 | 0.82 | 0.46 | 0.41 | 54.0% |
| `400_items.csv` | 400 | 0.82 | 0.46 | 0.41 | 51.7% |
| `600_items.csv` | 600 | 0.80 | 0.45 | 0.41 | 51.3% |

## 4. Spatial Diversity & Dimensional Realism
The density plots and table below quantify the generative quality using Wasserstein Distance—a measure of how closely the GAN's synthetic distribution matches the physical reality.

### 4.1 Distributional Fidelity Summary
Comparing Gaussian density overlaps and statistical moments between real and synthetic data.

| Dimension | Real Mean (μ) | GAN Mean (μ) | Real Std (σ) | GAN Std (σ) | Wasserstein Dist |
|:---|:---:|:---:|:---:|:---:|:---:|
| Item Length | 0.443 | 0.449 | 0.108 | 0.109 | **0.00921** |
| Item Width | 0.248 | 0.250 | 0.069 | 0.067 | **0.00351** |
| Item Height | 0.227 | 0.228 | 0.053 | 0.051 | **0.00351** |
| Item Weight | 6.827 | 6.856 | 2.579 | 2.601 | **0.09066** |

> **Note**: A lower Wasserstein Distance indicates higher distributional realism.

### 4.2 Generation Lifecycle Visual
![SKU Diversity Comparison](metrics_visuals/sku_diversity_comparison_full.png)

### Pipeline Reliability & Synthetic Diversity
This table provides a 4-way comparison of 5 random item samples tracking the synthetic lifecycle: from a real-world reference to the GAN's raw output, its physical reconstruction, and finally the scaled version used in training.

| Sample | Original (Real) | GAN Normalized [0-1] | GAN Denormalized | Synthetic (2x Scaled) |
|:---|:---|:---|:---|:---|
| 1 | (0.59, 0.20, 0.21, 7.7) | (0.604, 0.140, 0.036, 0.090) | (0.47, 0.14, 0.08, 2.9) | (0.94, 0.29, 0.16, 5.8) |
| 2 | (0.55, 0.28, 0.11, 8.4) | (0.221, 0.421, 0.216, 0.260) | (0.23, 0.23, 0.29, 7.2) | (0.47, 0.47, 0.58, 14.3) |
| 3 | (0.55, 0.28, 0.11, 8.4) | (0.484, 0.933, 0.122, 0.194) | (0.40, 0.40, 0.18, 5.5) | (0.79, 0.80, 0.36, 11.1) |
| 4 | (0.49, 0.13, 0.21, 5.1) | (0.477, 0.517, 0.193, 0.394) | (0.39, 0.27, 0.26, 10.5) | (0.78, 0.53, 0.53, 21.0) |
| 5 | (0.49, 0.13, 0.21, 5.1) | (0.710, 0.517, 0.170, 0.406) | (0.53, 0.27, 0.24, 10.8) | (1.07, 0.53, 0.48, 21.6) |

*(Format: Length, Width, Height, Weight)*


## 5. Phase-Based Sample Fidelity Dashboard (Summary)
This section compares 5 random samples across the three major generation phases. Full metadata is provided at the source phase.

### Phase 1: Original (Real-World Source)
| Smp | Len (m) | Wid (m) | Hei (m) | Wgt (kg) | Category | Fragile | Stack | Rotate |
|:---| :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| 1 | 0.590 | 0.200 | 0.210 | 7.67 | bakery products | False | True | True |
| 2 | 0.550 | 0.280 | 0.110 | 8.40 | confectionery | False | True | True |
| 3 | 0.550 | 0.280 | 0.110 | 8.40 | confectionery | False | True | True |
| 4 | 0.490 | 0.130 | 0.210 | 5.11 | candy | False | True | True |
| 5 | 0.490 | 0.130 | 0.210 | 5.11 | candy | False | True | True |


### Phase 2: GAN Latent Space (Normalized [0, 1])
| Smp | Item_L | Item_W | Item_H | Item_Wt | Data Type | Range |
|:---| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.603971 | 0.140313 | 0.036442 | 0.089580 | float32 | [0.0, 1.0] |
| 2 | 0.220504 | 0.421267 | 0.216233 | 0.259743 | float32 | [0.0, 1.0] |
| 3 | 0.484096 | 0.933467 | 0.121579 | 0.194445 | float32 | [0.0, 1.0] |
| 4 | 0.477372 | 0.516817 | 0.192554 | 0.394176 | float32 | [0.0, 1.0] |
| 5 | 0.709922 | 0.517451 | 0.170318 | 0.405620 | float32 | [0.0, 1.0] |


### Phase 3: GAN Denormalized (Reconstructed Source)
| Smp | Rec_Len | Rec_Wid | Rec_Hei | Rec_Wgt | Data Type | Unit |
|:---| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.468 | 0.145 | 0.082 | 2.91 | float32 | Physical |
| 2 | 0.235 | 0.235 | 0.291 | 7.15 | float32 | Physical |
| 3 | 0.395 | 0.399 | 0.181 | 5.53 | float32 | Physical |
| 4 | 0.391 | 0.265 | 0.263 | 10.50 | float32 | Physical |
| 5 | 0.533 | 0.266 | 0.238 | 10.79 | float32 | Physical |

---
