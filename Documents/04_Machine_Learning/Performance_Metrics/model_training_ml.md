# ML Model Training & Logic Report

> Auto-generated on **2026-03-31 18:45**

---

## 1. High-Intensity Hyperparameters
The following parameters were utilized to ensure robust convergence and positive R² across all 4 variants.

| Parameter | Value | Description |
|:--- |:--- |:--- |
| **Epochs (Full)** | 120 | Maximum training epochs for EO, GA, GA_EO variants |
| **Epochs (EO_GA)** | 40 | Reduced epochs for fast EO_GA variant |
| **Batch Size** | 2048 | Samples per GPU update |
| **Learning Rate** | 0.0005 | AdamW optimizer initial step size |
| **Optimizer** | AdamW | Weight decay=1e-4, with warmup+cosine LR schedule |
| **Spatial Weights** | X:[3,3] / EO_GA:[2,2] | Moderate spatial boost for stable R² |
| **Patience (Full)** | 20 | Early stopping patience for full models |
| **Patience (EO_GA)** | 8 | Aggressive early-stop for EO_GA speed |
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
| `EO` | 0.145702 | 25.05% | **Terminated @ Ep 58** | 0.0000 |
| `EO_GA` | 0.105753 | 25.11% | Converged Naturally | 0.0000 |
| `GA` | 0.145679 | 24.99% | **Terminated @ Ep 47** | 0.0000 |
| `GA_EO` | 0.145661 | 25.11% | **Terminated @ Ep 57** | 0.0000 |

## 5. Hardware & System Context
- **CPU**: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- **GPU**: NVIDIA GeForce RTX 3060
- **RAM**: 47.91 GB
- **Datasets**: 500,000 Total Synthetic Rows (125k Shared Master)