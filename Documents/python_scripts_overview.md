# Python Scripts Overview

This document provides a comprehensive overview of the Python scripts used in the Warehouse Bin-Packing project, detailing their specific roles within the pipeline.

---

## 🏗️ Core Application & Infrastructure

### app.py
The central **Flask API server**. It serves the web interface and handles all backend logic, including:
- **Warehouse Management**: CRUD operations for warehouse configurations and exclusion zones.
- **Item Management**: CSV upload/export, item scrambling, and manual item entry.
- **Optimization Orchestration**: Triggers various optimization algorithms (GA, EO, Hybrid) in background threads and tracks their progress.
- **Data Integration**: Connects the frontend to the database and the ML inference engine.

### database.py
The **SQLite Database Interface**. It manages the local `warehouse.db` file and provides helper functions for persistent storage:
- **Schema Management**: Initializes tables for items, warehouses, exclusion zones, and optimization results.
- **Migrations**: Handles database updates when schema changes occur.
- **CRUD Helpers**: Abstracted functions to interact with the database without writing raw SQL in other scripts.

---

## 🧠 Machine Learning & Optimization

### ml_utils.py
Contains the **Neural Network Architecture** and ML helper classes:
- **`PackingModel`**: A deep neural network (PyTorch) designed to predict optimal (x, y, z, rotation) coordinates for items based on 18 spatial and geometric features.
- **`MLOptimizer`**: A high-level class that loads trained models and performs inference to guide the packing process.

### optimizer.py
The **Core Packing Engine**. It implements the mathematical logic for item placement:
- **Heuristics**: Contains the Genetic Algorithm (GA) and Extremal Optimization (EO) logic, though these are now primarily used to generate training data for the ML models.
- **`repair_solution_compact`**: A critical heuristic that takes raw predictions (from GA or ML) and "settles" them using gravity, ensuring no overlaps and strict adherence to stacking/fragility constraints.
- **Spatial Grid**: Uses a `SimpleGrid` class to provide $O(1)$ spatial lookups, significantly accelerating collision detection.

### optimizer_physics.py
An **Optional Physics Refinement** script:
- Uses **PyBullet** (a 3D physics engine) to simulate gravity and collisions.
- It can be used as a final "settlement" step to ensure items are perfectly aligned and stable.

---

## 📉 Training & Evaluation Pipeline

### train_models.py
The main **Model Training Script**:
- Trains the 4 different model variants (GA, EO, and Hybrids) using the data generated in `training_data/`.
- Features early stopping and learning rate scheduling (Cosine Annealing) to ensure high-quality spatial predictions.

### evaluate_metrics.py
The **Performance Audit Tool**:
- Re-trains models and evaluates them against unseen GAN-generated datasets (200, 400, and 600 items).
- Calculates deep logistics metrics like **Bounding Box Efficiency**, **Center of Gravity (CoG)**, and **Fragility Compliance**.
- Generates the comprehensive `MODEL_METRICS.md` report.

### generate_training_data.py
The **Synthetic Data Generator**:
- Uses the GAN to create realistic item sets.
- Runs the `repair_solution_compact` heuristic in different "Dense" vs "Normal" scenarios to create 50,000+ rows of training data per model variant.

---

## 🧪 GAN (Synthetic Data)

### gan/model.py
Defines the **Generative Adversarial Network** architecture (Generator and Discriminator) used to learn and replicate item dimension distributions from the real-world dataset.

### gan/train.py
The training loop for the GAN. It learns from the `datasets/datasets.csv` file to produce realistic synthetic item dimensions.

### gan/generate.py
A CLI tool to use the trained GAN to output specific quantities of synthetic items into CSV files (e.g., `200_items.csv`).

### gan/data_loader.py
Provides data transformations and normalization needed to feed real-world item data into the GAN training process.

---

## 🛠️ Utilities

### datasets/convert_dataset.py
A data cleaning script that converts raw JSON data (like `bed-bpp_v1.json`) into the standard `datasets.csv` format used by the rest of the system.

### test_script.py
A simple utility for one-off testing and debugging of specific system components.
