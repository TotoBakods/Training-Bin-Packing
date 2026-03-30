# GAN Performance & Generation Report

> Auto-generated on **2026-03-31 05:53**

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
| `dataset.csv` | 428720 | 0.44 | 0.25 | 0.23 | 51.5% |

## 3. Original Dataset Reference (First 5 Items)
The following table shows the raw source data from `datasets.csv` used for GAN training:

| id | name | length | width | height | weight | category | priority | fragility | stackable | access_freq | can_rotate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 00103095 | ciabatta-00103095 | 0.59 | 0.2 | 0.21 | 7.67 | bakery products | 1 | 0 | 1 | 1 | 1 |
| 00111025 | cake-00111025 | 0.55 | 0.28 | 0.11 | 8.4 | confectionery | 1 | 0 | 1 | 1 | 1 |
| 00111025 | cake-00111025 | 0.55 | 0.28 | 0.11 | 8.4 | confectionery | 1 | 0 | 1 | 1 | 1 |
| 00104636 | dessert-00104636 | 0.49 | 0.13 | 0.21 | 5.11 | candy | 1 | 0 | 1 | 1 | 1 |
| 00104636 | dessert-00104636 | 0.49 | 0.13 | 0.21 | 5.11 | candy | 1 | 0 | 1 | 1 | 1 |

## 4. Spatial Diversity & Dimensional Realism
The density plots and table below quantify the generative quality using Wasserstein Distance—a measure of how closely the GAN's synthetic distribution matches the physical reality.

### 4.1 Distributional Fidelity Summary
Comparing Gaussian density overlaps and statistical moments between real and synthetic data.

| Dimension | Real Mean (μ) | GAN Mean (μ) | Real Std (σ) | GAN Std (σ) | Wasserstein Dist |
|:---|:---:|:---:|:---:|:---:|:---:|
| Item Length | 0.443 | 0.443 | 0.108 | 0.109 | **0.00450** |
| Item Width | 0.248 | 0.248 | 0.069 | 0.067 | **0.00353** |
| Item Height | 0.227 | 0.229 | 0.053 | 0.050 | **0.00393** |
| Item Weight | 6.827 | 6.754 | 2.579 | 2.626 | **0.11882** |

> **Note**: A lower Wasserstein Distance indicates higher distributional realism.

### 4.2 Generation Lifecycle Visual
![SKU Diversity Comparison](metrics_visuals/sku_diversity_comparison_full.png)


## 5. Sample Fidelity Matrix (Lifecycle Dashboard)
This dashboard tracks 5 random samples through the generation lifecycle. For each sample, we compare the **Original** (Real) source, the **Latent** (Normalized GAN output), and the **Reconstructed** (Denormalized) result.

| Sample | Phase | Length | Width | Height | Weight | Metadata / Attributes |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **#1** | **Original** | 0.590m | 0.200m | 0.210m | 7.67kg | **Cat**: bakery products, **F/S/R**: N/Y/Y |
| | GAN Latent | 0.8035 | 0.9190 | 0.2095 | 0.2213 | [0.0, 1.0] Range |
| | Reconst. | 0.590m | 0.394m | 0.283m | 6.19kg | Physical Units |
| --- | --- | --- | --- | --- | --- | --- |
| **#2** | **Original** | 0.550m | 0.280m | 0.110m | 8.40kg | **Cat**: confectionery, **F/S/R**: N/Y/Y |
| | GAN Latent | 0.2756 | 0.3499 | 0.1330 | 0.0843 | [0.0, 1.0] Range |
| | Reconst. | 0.268m | 0.212m | 0.194m | 2.78kg | Physical Units |
| --- | --- | --- | --- | --- | --- | --- |
| **#3** | **Original** | 0.550m | 0.280m | 0.110m | 8.40kg | **Cat**: confectionery, **F/S/R**: N/Y/Y |
| | GAN Latent | 0.3074 | 0.3304 | 0.1359 | 0.2042 | [0.0, 1.0] Range |
| | Reconst. | 0.288m | 0.206m | 0.198m | 5.77kg | Physical Units |
| --- | --- | --- | --- | --- | --- | --- |
| **#4** | **Original** | 0.490m | 0.130m | 0.210m | 5.11kg | **Cat**: candy, **F/S/R**: N/Y/Y |
| | GAN Latent | 0.7392 | 0.3893 | 0.1552 | 0.2445 | [0.0, 1.0] Range |
| | Reconst. | 0.551m | 0.225m | 0.220m | 6.77kg | Physical Units |
| --- | --- | --- | --- | --- | --- | --- |
| **#5** | **Original** | 0.490m | 0.130m | 0.210m | 5.11kg | **Cat**: candy, **F/S/R**: N/Y/Y |
| | GAN Latent | 0.2705 | 0.1965 | 0.1388 | 0.0871 | [0.0, 1.0] Range |
| | Reconst. | 0.265m | 0.163m | 0.201m | 2.85kg | Physical Units |

---
