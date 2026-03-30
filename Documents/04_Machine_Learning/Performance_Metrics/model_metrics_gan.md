# GAN Performance & Generation Report

> Auto-generated on **2026-03-31 05:28**

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
| Item Length | 0.443 | 0.447 | 0.108 | 0.108 | **0.00744** |
| Item Width | 0.248 | 0.249 | 0.069 | 0.068 | **0.00430** |
| Item Height | 0.227 | 0.230 | 0.053 | 0.050 | **0.00410** |
| Item Weight | 6.827 | 6.793 | 2.579 | 2.598 | **0.08562** |

> **Note**: A lower Wasserstein Distance indicates higher distributional realism.

### 4.2 Generation Lifecycle Visual
![SKU Diversity Comparison](metrics_visuals/sku_diversity_comparison_full.png)

### Pipeline Reliability & Synthetic Diversity
This table provides a 4-way comparison of 5 random item samples tracking the synthetic lifecycle: from a real-world reference to the GAN's raw output, its physical reconstruction, and finally the scaled version used in training.

| Sample | Original (Real) | GAN Normalized [0-1] | GAN Denormalized | Synthetic (2x Scaled) |
|:---|:---|:---|:---|:---|
| 1 | (0.59, 0.20, 0.21, 7.7) | (0.383, 0.322, 0.213, 0.196) | (0.33, 0.20, 0.29, 5.6) | (0.67, 0.41, 0.57, 11.1) |
| 2 | (0.55, 0.28, 0.11, 8.4) | (0.479, 0.926, 0.126, 0.159) | (0.39, 0.40, 0.19, 4.6) | (0.78, 0.79, 0.37, 9.3) |
| 3 | (0.55, 0.28, 0.11, 8.4) | (0.382, 0.336, 0.216, 0.187) | (0.33, 0.21, 0.29, 5.3) | (0.67, 0.42, 0.58, 10.7) |
| 4 | (0.49, 0.13, 0.21, 5.1) | (0.447, 0.401, 0.200, 0.207) | (0.37, 0.23, 0.27, 5.8) | (0.75, 0.46, 0.54, 11.7) |
| 5 | (0.49, 0.13, 0.21, 5.1) | (0.263, 0.191, 0.139, 0.083) | (0.26, 0.16, 0.20, 2.7) | (0.52, 0.32, 0.40, 5.5) |

*(Format: Length, Width, Height, Weight)*


## 5. Sample Fidelity Dashboard (Compact Matrix)
This section maps the transformation of 5 random samples from their physical source to the latent model space and back to the reconstructed synthetic item.

### Sample 1 Fidelity Trace
| Attribute | Real Value | Latent [0-1] | Reconstructed | Type / Unit |
|:---|:---:|:---:|:---:|:---|
| **Length**     | 0.590 | 0.382591 | 0.333 | f32 / meters |
| **Width**      | 0.200 | 0.322347 | 0.203 | f32 / meters |
| **Height**     | 0.210 | 0.212936 | 0.287 | f32 / meters |
| **Weight**     | 7.670 | 0.196205 | 5.569 | f32 / kg     |
| Category       | bakery products | -- | -- | obj / str    |
| Fragility      | False | -- | -- | i64 / bool   |
| Stackable      | True | -- | -- | i64 / bool   |
| Can Rotate     | True | -- | -- | i64 / bool   |

---

### Sample 2 Fidelity Trace
| Attribute | Real Value | Latent [0-1] | Reconstructed | Type / Unit |
|:---|:---:|:---:|:---:|:---|
| **Length**     | 0.550 | 0.479437 | 0.392 | f32 / meters |
| **Width**      | 0.280 | 0.926063 | 0.396 | f32 / meters |
| **Height**     | 0.110 | 0.125990 | 0.186 | f32 / meters |
| **Weight**     | 8.400 | 0.158898 | 4.640 | f32 / kg     |
| Category       | confectionery | -- | -- | obj / str    |
| Fragility      | False | -- | -- | i64 / bool   |
| Stackable      | True | -- | -- | i64 / bool   |
| Can Rotate     | True | -- | -- | i64 / bool   |

---

### Sample 3 Fidelity Trace
| Attribute | Real Value | Latent [0-1] | Reconstructed | Type / Unit |
|:---|:---:|:---:|:---:|:---|
| **Length**     | 0.550 | 0.381959 | 0.333 | f32 / meters |
| **Width**      | 0.280 | 0.336348 | 0.208 | f32 / meters |
| **Height**     | 0.110 | 0.215530 | 0.290 | f32 / meters |
| **Weight**     | 8.400 | 0.186649 | 5.331 | f32 / kg     |
| Category       | confectionery | -- | -- | obj / str    |
| Fragility      | False | -- | -- | i64 / bool   |
| Stackable      | True | -- | -- | i64 / bool   |
| Can Rotate     | True | -- | -- | i64 / bool   |

---

### Sample 4 Fidelity Trace
| Attribute | Real Value | Latent [0-1] | Reconstructed | Type / Unit |
|:---|:---:|:---:|:---:|:---|
| **Length**     | 0.490 | 0.447398 | 0.373 | f32 / meters |
| **Width**      | 0.130 | 0.400756 | 0.228 | f32 / meters |
| **Height**     | 0.210 | 0.199667 | 0.272 | f32 / meters |
| **Weight**     | 5.110 | 0.207451 | 5.850 | f32 / kg     |
| Category       | candy | -- | -- | obj / str    |
| Fragility      | False | -- | -- | i64 / bool   |
| Stackable      | True | -- | -- | i64 / bool   |
| Can Rotate     | True | -- | -- | i64 / bool   |

---

### Sample 5 Fidelity Trace
| Attribute | Real Value | Latent [0-1] | Reconstructed | Type / Unit |
|:---|:---:|:---:|:---:|:---|
| **Length**     | 0.490 | 0.262661 | 0.260 | f32 / meters |
| **Width**      | 0.130 | 0.190584 | 0.161 | f32 / meters |
| **Height**     | 0.210 | 0.138657 | 0.201 | f32 / meters |
| **Weight**     | 5.110 | 0.082635 | 2.739 | f32 / kg     |
| Category       | candy | -- | -- | obj / str    |
| Fragility      | False | -- | -- | i64 / bool   |
| Stackable      | True | -- | -- | i64 / bool   |
| Can Rotate     | True | -- | -- | i64 / bool   |

---
