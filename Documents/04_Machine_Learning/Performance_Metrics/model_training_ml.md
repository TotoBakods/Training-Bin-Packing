# ML Model Training & Logic Report

> Auto-generated on **2026-04-02 14:51**

---

## 1. High-Intensity Hyperparameters
The following parameters were utilized to ensure robust convergence and positive R² across all 4 variants. Each algorithm's personality is reflected in these settings.

| Parameter | Standalone GA/EO | GA-EO / EO-GA Hybrid | Description |
|:--- |:---: |:---: |:--- |
| **Epochs** | 120 | 100 | Training iterations (EO-GA prioritized for speed) |
| **Batch Size** | 2048 | 2048 | Samples per GPU update |
| **Learning Rate** | 0.0005 | 0.0005 | AdamW optimizer initial step size |
| **Spatial Weights** | X:3.0, Y:3.0 | X:2.0, Y:2.0 | Spatial boost for stable R² |
| **Patience** | 20 | 15 | Early stopping threshold |
| **Collision Weight** | 1.5 | 1.0 | Physics-aware loss penalty factor |
## 2. Training Convergence Progression
The models were trained on 125,000 synthetic samples per variant. The objective is to minimize spatial prediction error while maximizing fitness.

### Fitness (Validation R²) Progression
![Fitness Curves](metrics_visuals/training_fitness_curves.png)

### Training & Validation Loss
![Loss Grid](metrics_visuals/training_loss_curves.png)


## 3. Heuristic Design Optimization
- **Execution Efficiency**: Reduced search space attempts to **20 per item**, resulting in a significant reduction in overall repair latency.
- **Selective Convergence**: The EO_GA variant utilizes targeted early-stopping to prevent over-fitting while maintaining high throughput.

## 4. Heuristic Variant Performance & Logic
| Model Variant | Final Loss | Final Fitness (%) | Early Stop Log | Stability (PyBullet) |
|:--- |:---: |:---: |:--- |:---: |
| `EO` | 0.145779 | 24.99% | **Terminated @ Ep 55** | 0.0000 |
| `EO_GA` | 0.105846 | 25.13% | **Terminated @ Ep 56** | 0.0000 |
| `GA` | 0.145657 | 25.07% | **Terminated @ Ep 54** | 0.0000 |
| `GA_EO` | 0.145554 | 25.10% | **Terminated @ Ep 40** | 0.0000 |

## 5. Hardware & System Context
- **CPU**: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- **GPU**: NVIDIA GeForce RTX 3060
- **RAM**: 47.91 GB
- **Datasets**: 500,000 Total Synthetic Rows (125k Shared Master)