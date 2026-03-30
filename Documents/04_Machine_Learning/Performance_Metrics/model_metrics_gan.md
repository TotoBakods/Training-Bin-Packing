# GAN Performance & Generation Report

> Auto-generated on **2026-03-31 05:33**

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
| Item Length | 0.443 | 0.445 | 0.108 | 0.110 | **0.00684** |
| Item Width | 0.248 | 0.247 | 0.069 | 0.068 | **0.00364** |
| Item Height | 0.227 | 0.228 | 0.053 | 0.050 | **0.00405** |
| Item Weight | 6.827 | 6.797 | 2.579 | 2.665 | **0.11774** |

> **Note**: A lower Wasserstein Distance indicates higher distributional realism.

### 4.2 Generation Lifecycle Visual
![SKU Diversity Comparison](metrics_visuals/sku_diversity_comparison_full.png)

### Pipeline Reliability & Synthetic Diversity
This table provides a 4-way comparison of 5 random item samples tracking the synthetic lifecycle: from a real-world reference to the GAN's raw output, its physical reconstruction, and finally the scaled version used in training.

| Sample | Original (Real) | GAN Normalized [0-1] | GAN Denormalized | Synthetic (2x Scaled) |
|:---|:---|:---|:---|:---|
| 1 | (0.59, 0.20, 0.21, 7.7) | (0.415, 0.314, 0.188, 0.145) | (0.35, 0.20, 0.26, 4.3) | (0.71, 0.40, 0.52, 8.6) |
| 2 | (0.55, 0.28, 0.11, 8.4) | (0.361, 0.276, 0.216, 0.151) | (0.32, 0.19, 0.29, 4.5) | (0.64, 0.38, 0.58, 8.9) |
| 3 | (0.55, 0.28, 0.11, 8.4) | (0.380, 0.278, 0.218, 0.172) | (0.33, 0.19, 0.29, 5.0) | (0.66, 0.38, 0.58, 9.9) |
| 4 | (0.49, 0.13, 0.21, 5.1) | (0.468, 0.577, 0.150, 0.162) | (0.39, 0.28, 0.21, 4.7) | (0.77, 0.57, 0.43, 9.4) |
| 5 | (0.49, 0.13, 0.21, 5.1) | (0.799, 0.308, 0.150, 0.246) | (0.59, 0.20, 0.21, 6.8) | (1.17, 0.40, 0.43, 13.6) |

*(Format: Length, Width, Height, Weight)*


## 5. Sample Fidelity Dashboard (Organized)
This section tracks 5 random items throughout the synthetic lifecycle, with attributes grouped by physical dimensions and SKU metadata.

### Sample 1 Fidelity Profile
| Component | Attribute | Original (Real) | GAN Latent [0-1] | GAN Denormalized | Metadata |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Physical** | Length     | 0.590 | 0.415037 | 0.353 | meters (f32) |
| **Physical** | Width      | 0.200  | 0.313882 | 0.200 | meters (f32) |
| **Physical** | Height     | 0.210 | 0.187908 | 0.258 | meters (f32) |
| **Physical** | Weight     | 7.670 | 0.145461 | 4.305 | kg (f32)     |
| **SKU Meta** | Category   | bakery products   | -- | -- | object (str) |
| **SKU Meta** | Fragility  | False | -- | -- | int64 (bool) |
| **SKU Meta** | Stackable  | True | -- | -- | int64 (bool) |
| **SKU Meta** | Can Rotate | True | -- | -- | int64 (bool) |

---

### Sample 2 Fidelity Profile
| Component | Attribute | Original (Real) | GAN Latent [0-1] | GAN Denormalized | Metadata |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Physical** | Length     | 0.550 | 0.360815 | 0.320 | meters (f32) |
| **Physical** | Width      | 0.280  | 0.275562 | 0.188 | meters (f32) |
| **Physical** | Height     | 0.110 | 0.216027 | 0.291 | meters (f32) |
| **Physical** | Weight     | 8.400 | 0.151292 | 4.450 | kg (f32)     |
| **SKU Meta** | Category   | confectionery   | -- | -- | object (str) |
| **SKU Meta** | Fragility  | False | -- | -- | int64 (bool) |
| **SKU Meta** | Stackable  | True | -- | -- | int64 (bool) |
| **SKU Meta** | Can Rotate | True | -- | -- | int64 (bool) |

---

### Sample 3 Fidelity Profile
| Component | Attribute | Original (Real) | GAN Latent [0-1] | GAN Denormalized | Metadata |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Physical** | Length     | 0.550 | 0.380225 | 0.332 | meters (f32) |
| **Physical** | Width      | 0.280  | 0.278101 | 0.189 | meters (f32) |
| **Physical** | Height     | 0.110 | 0.217579 | 0.292 | meters (f32) |
| **Physical** | Weight     | 8.400 | 0.171955 | 4.965 | kg (f32)     |
| **SKU Meta** | Category   | confectionery   | -- | -- | object (str) |
| **SKU Meta** | Fragility  | False | -- | -- | int64 (bool) |
| **SKU Meta** | Stackable  | True | -- | -- | int64 (bool) |
| **SKU Meta** | Can Rotate | True | -- | -- | int64 (bool) |

---

### Sample 4 Fidelity Profile
| Component | Attribute | Original (Real) | GAN Latent [0-1] | GAN Denormalized | Metadata |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Physical** | Length     | 0.490 | 0.467556 | 0.385 | meters (f32) |
| **Physical** | Width      | 0.130  | 0.577093 | 0.285 | meters (f32) |
| **Physical** | Height     | 0.210 | 0.150028 | 0.214 | meters (f32) |
| **Physical** | Weight     | 5.110 | 0.162233 | 4.723 | kg (f32)     |
| **SKU Meta** | Category   | candy   | -- | -- | object (str) |
| **SKU Meta** | Fragility  | False | -- | -- | int64 (bool) |
| **SKU Meta** | Stackable  | True | -- | -- | int64 (bool) |
| **SKU Meta** | Can Rotate | True | -- | -- | int64 (bool) |

---

### Sample 5 Fidelity Profile
| Component | Attribute | Original (Real) | GAN Latent [0-1] | GAN Denormalized | Metadata |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Physical** | Length     | 0.490 | 0.799002 | 0.587 | meters (f32) |
| **Physical** | Width      | 0.130  | 0.308023 | 0.199 | meters (f32) |
| **Physical** | Height     | 0.210 | 0.149714 | 0.214 | meters (f32) |
| **Physical** | Weight     | 5.110 | 0.246071 | 6.812 | kg (f32)     |
| **SKU Meta** | Category   | candy   | -- | -- | object (str) |
| **SKU Meta** | Fragility  | False | -- | -- | int64 (bool) |
| **SKU Meta** | Stackable  | True | -- | -- | int64 (bool) |
| **SKU Meta** | Can Rotate | True | -- | -- | int64 (bool) |

---
