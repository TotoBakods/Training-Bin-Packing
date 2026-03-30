# ML Model Training & Benchmarking Report

> Auto-generated on **2026-03-31 03:11**

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
| `FIT_EO_GA` | 1.0000 | 0.0000 | 0.0000 |
### Modern Performance Optimizations
- **Spatial Grid ($O(1)$)**: Initialized `SimpleGrid` for constant-time neighbor collision checks.
- **Early-Exit Logic**: Search terminates immediately if `z=0` (floor positioning) is achieved.
- **Search Pruning**: Successfully reduced search attempts from 50 to 20 without increasing placement collisions.

### Physical Validity Proof (PyBullet Settlement)
The heatmap below visualizes the average settlement displacement across the warehouse floor. Regions in **red** indicate areas where the heuristic label predicted placements that required significant physical correction.

![Stability Heatmap](metrics_visuals/stability_heatmap.png)


## 3. Training Convergence & Loss Logs
![Loss Grid](metrics_visuals/training_loss_curves.png)

| Model | Final Train MSE | Final Val MSE | Overfit Gap |
|-------|-----------------|---------------|-------------|
| `model_fit_eo_ga` | 0.079976 | 0.081009 | +0.001034 |

## 4. Final Inference Benchmarking
Benchmarks across varied workload sizes (200, 400, 600 items).

### Dataset: `200_items.csv`
| Model | Fitness | Space % | Time (s) |
|-------|---------|---------|----------|
| `model_fit_eo_ga` | 0.30 | 1.42% | 4.85s |

### Dataset: `400_items.csv`
| Model | Fitness | Space % | Time (s) |
|-------|---------|---------|----------|
| `model_fit_eo_ga` | 0.30 | 2.81% | 14.57s |

### Dataset: `600_items.csv`
| Model | Fitness | Space % | Time (s) |
|-------|---------|---------|----------|
| `model_fit_eo_ga` | 0.22 | 4.14% | 31.44s |
