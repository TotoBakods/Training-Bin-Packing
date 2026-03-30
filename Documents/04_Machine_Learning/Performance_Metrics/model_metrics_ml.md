# ML Model Training & Benchmarking Report

> Auto-generated on **2026-03-31 02:17**

---

## 1. Training Architecture & System Logs

### Hardware Context
- **Hardware**: NVIDIA GeForce RTX 3060
- **Memory**: 47.91 GB

### Model Hyperparameters
- **Training Epochs**: 200
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Validation Split**: 20%

---

## 2. Physics Settlement Verification (Training Data Proof)
Representative scenarios from the training sets were simulated in PyBullet to verify label stability.

| Variant | Stability Index | Mean Displacement (m) | Max Displacement (m) |
|---------|-----------------|-----------------------|----------------------|
| `FIT_EO` | 1.0000 | 0.0000 | 0.0000 |
| `FIT_EO_GA` | 1.0000 | 0.0000 | 0.0000 |
| `FIT_GA` | 1.0000 | 0.0000 | 0.0000 |
| `FIT_GA_EO` | 1.0000 | 0.0000 | 0.0000 |

### Spatial Stability Distribution (Heatmap)
The heatmap below visualizes the average settlement displacement across the warehouse floor. Regions in **red** indicate areas where the heuristic label predicted placements that required significant physical correction.

![Stability Heatmap](metrics_visuals/stability_heatmap.png)


## 3. Training Convergence & Loss Logs
![Loss Grid](metrics_visuals/training_loss_curves.png)

| Model | Final Train MSE | Final Val MSE | Overfit Gap |
|-------|-----------------|---------------|-------------|
| `model_fit_eo` | 0.080063 | 0.082357 | +0.002294 |
| `model_fit_eo_ga` | 0.079891 | 0.081103 | +0.001212 |
| `model_fit_ga` | 0.079553 | 0.081332 | +0.001779 |
| `model_fit_ga_eo` | 0.080093 | 0.080262 | +0.000169 |

## 4. Final Inference Benchmarking
Benchmarks across varied workload sizes (200, 400, 600 items).

### Dataset: `200_items.csv`
| Model | Fitness | Space % | Time (s) |
|-------|---------|---------|----------|
| `model_fit_eo` | 0.30 | 1.42% | 4.36s |
| `model_fit_eo_ga` | 0.30 | 1.42% | 4.63s |
| `model_fit_ga` | 0.30 | 1.42% | 4.42s |
| `model_fit_ga_eo` | 0.30 | 1.42% | 4.14s |

### Dataset: `400_items.csv`
| Model | Fitness | Space % | Time (s) |
|-------|---------|---------|----------|
| `model_fit_eo` | 0.30 | 2.81% | 13.27s |
| `model_fit_eo_ga` | 0.30 | 2.81% | 13.71s |
| `model_fit_ga` | 0.30 | 2.81% | 13.16s |
| `model_fit_ga_eo` | 0.30 | 2.81% | 13.53s |

### Dataset: `600_items.csv`
| Model | Fitness | Space % | Time (s) |
|-------|---------|---------|----------|
| `model_fit_eo` | 0.22 | 4.14% | 27.55s |
| `model_fit_eo_ga` | 0.23 | 4.14% | 27.93s |
| `model_fit_ga` | 0.23 | 4.14% | 28.06s |
| `model_fit_ga_eo` | 0.22 | 4.14% | 27.17s |
