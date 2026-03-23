# Core Logistics - Training Bin Packing

## Project Overview
This project is an advanced web-based warehouse management and 3D bin packing optimization system. Its primary purpose is to find the optimal spatial arrangement of items within a warehouse in order to maximize space utilization, accessibility, and stability. 

Instead of traditional heuristic approaches like Genetic Algorithms (GA) or Extremal Optimization (EO), it uses machine learning models (trained via imitation learning) to predict the optimal layout. Once the AI model predicts the layout, a heuristic repair function and a rigid-body physics engine (PyBullet) settle the items to ensure a realistic, stable configuration. The project also provides a rich 3D visualization interface to view and interact with the warehouse layout.

---

## File Descriptions

| File Name & Path | Role & Responsibility | Key Components, Classes, & Modules | Formulas, Algorithms, or Logic Used |
| :--- | :--- | :--- | :--- |
| `app.py` | **Backend API & Entry Point**<br>Serves the frontend, provides REST endpoints, and manages optimization tasks. | - Routes: `/api/items`, `/api/optimize/*`, `/api/upload-csv`<br>- `finalize_optimization()`: Connects ML output to the physics engine and saves results to DB.<br>- Global state dictionaries track long-running optimization threads. | **Concurrency:** Uses `threading.Thread` to run ML optimization without blocking the Flask API. |
| `database.py` | **Data Persistence**<br>SQLite database manager for items, warehouses, zones, and results. | - Tables: `items`, `warehouse_config`, `exclusion_zones`, `optimization_results`<br>- Functions: `get_all_items()`, `save_solution()`, `add_warehouse()`, `load_sample_data()` | **CRUD Logic:** Executes SQLite queries for fast, local data retrieval and storage. |
| `ml_utils.py` | **Machine Learning Inference**<br>Loads PyTorch models and maps warehouse data into tensors for predictions. | - `PackingModel`: PyTorch Neural Network (`nn.Module`) with 4 hidden layers estimating (x,y,z,rot).<br>- `MLOptimizer`: Prepares features, runs model inference, and calls heuristic repairs. | **Neural Network:** A feed-forward deep learning model with ReLU activations that outputs normalized coordinates. |
| `optimizer.py` | **Heuristics & Fitness Evaluation**<br>Evaluates solutions and implements logic to repair strictly invalid AI layouts. | - `fitness_function_numpy()`: Computes multi-objective fitness scores.<br>- `repair_solution_compact()`: A gravity-based heuristic that adjusts coordinates to avoid mid-air floating.<br>- `calculate_z_for_item()`: Finds valid Z drops for stacking. | **AABB Overlap & Gravity:** Simulates basic collision detection and calculates the highest intersecting Z point beneath an item to calculate gravity. |
| `optimizer_physics.py` | **Physics Engine Settlement**<br>Runs a PyBullet physics simulation to settle the AI-predicted layout. | - `physics_settle()`: Creates ground planes, walls, and drops items based on mass to resolve overlapping bounds or instability. | **Rigid-Body Dynamics:** Uses PyBullet to simulate gravity, lateral friction (0.8), spinning friction (0.1), and realistic resting states over 1000 discrete time steps. |
| `train_models.py` | **Model Training Pipeline**<br>Trains the PyTorch Neural Networks on historical packing data. | - `WarehouseDataset`: Parses CSV training data into PyTorch `DataLoader`.<br>- `train_model()`: Uses Mean Squared Error (MSE) loss and Adam optimizer to train models to imitate algorithms like GA or EO. | **MSE Loss:** Computes the mean squared error between the model's predicted (x,y,z,rot) and the target ground truth. |
| `index.html` | **Frontend Layout**<br>Contains the HTML structure for the app. | Sidebar, Header, 3D viewport (`#three-container`), and Inspector panel. | **DOM Structure:** Provides the skeleton for data injection and 3D canvas rendering. |
| `script.js` | **Frontend Logic**<br>Manages the 3D rendering, API integration, and UI state. | - `renderItems()`: Visualizes bins in 3D using `Three.js`.<br>- `startOptimization()`: Triggers backend threads via `fetch`.<br>- `generatePickerPath()`: Renders the optimal picking route. | **Nearest-Neighbor TSP:** Uses a greedy nearest-neighbor algorithm to draw a picking path from the warehouse door to selected items. |
| `style.css` | **App Styling** | Modern CSS variables, flexbox grids, and UI themes. | **CSS Variables:** Implements theming using dynamic color palettes and flexbox layouts. |

---

## Formulas & Logic Reference

### 1. Space Utilization
* **What it computes:** The percentage of the total warehouse volume that is occupied by items.
* **Where it's used:** `optimizer.py` -> `fitness_function_numpy()`
```python
total_item_volume = sum([item.length * item.width * item.height for item in items])
warehouse_volume = wh_length * wh_width * wh_height
space_util = total_item_volume / warehouse_volume
```

### 2. Accessibility Score
* **What it computes:** Evaluates how easy it is to retrieve items, prioritizing frequently accessed items closer to the warehouse door. It uses an inverse distance weighting.
* **Where it's used:** `optimizer.py` -> `fitness_function_numpy()`
```python
# Calculates distance from door (door_x, door_y) for each item position (x, y)
dist = sqrt((x - door_x)**2 + (y - door_y)**2)
access_score = 1.0 / (1.0 + dist)

# Final accessibility is the weighted average based on access_frequency arrays
accessibility = average(access_scores, weights=access_frequencies)
```

### 3. Stability Checks & Gravity
* **What it computes:** Determines the valid resting Z-axis coordinate by checking bounding box overlaps of items beneath it. If an item is not supported by floor or at least 20% of its base on another item, it is unstable.
* **Where it's used:** `optimizer.py` -> `calculate_z_for_item()` and `fitness_function_numpy()`
```python
# Intersecting area between new item and supporting items beneath it
support_area = width(intersect) * depth(intersect)
if support_area > (item.length * item.width) * 0.2:
    # Item is marked as stable if > 20% of its base is supported by the items below it
```

### 4. Overlap Penalty (AABB Collision)
* **What it computes:** Axis-Aligned Bounding Box (AABB) collision detection function used to count overlaps between items and penalize invalid packing layouts during evaluation.
* **Where it's used:** `optimizer.py` -> `fitness_function_numpy()` (Vectorized via NumPy)
```python
# For two items with centers (x1,y1) and (x2,y2) and dimensions (w1,d1) and (w2,d2)
overlap_x = abs(x1 - x2) < (w1/2 + w2/2)
overlap_y = abs(y1 - y2) < (d1/2 + d2/2)
overlap_z = z_top1 > z_bottom2 and z_bottom1 < z_top2

is_colliding = overlap_x and overlap_y and overlap_z
```

---

## End-to-End Workflow

**1. Data Ingestion & Configuration**
* **Trigger:** The user opens the web app and uploads a CSV or uses the API to add items.
* **Stage Process:** `script.js` fetches warehouse configurations and current inventory from `app.py`. Items and constraints are saved and loaded from the SQLite DB (`database.py`).
* **Output:** The items are rendered in a 3D viewport natively in the browser before optimization.

**2. Optimization Triggered**
* **Trigger:** The user selects an algorithm (e.g., GA, EO) and desired weights (Space vs Accessibility vs Stability) and clicks "Initiate Optimization".
* **Stage Process:** `script.js` makes a background POST request to `/api/optimize/<algo>`.
* **Output:** The UI updates its status to "OPTIMIZING..." while waiting for the background thread to finish.

**3. Machine Learning Inference**
* **Trigger:** The API route receives the POST request.
* **Stage Process:** `app.py` spins up a background thread that invokes `MLOptimizer.optimize()` (`ml_utils.py`). The method processes all items into a feature matrix (normalizing dimensions and flags) and inputs it into a pre-trained PyTorch Neural Network (`PackingModel`).
* **Output:** The neural network outputs a raw sequence of floating-point predictions for `(x, y, z, rotation)`.

**4. Heuristic Repair & Constraints**
* **Trigger:** The raw ML predictions are generated and passed down the pipeline.
* **Stage Process:** The predictions are passed into `repair_solution_compact()` (`optimizer.py`). This function strictly enforces boundary walls, exclusion zones, and ensures minimal gravitational support (preventing items from floating mid-air or clipping outside boundaries).
* **Output:** A mathematically valid, but potentially physically imperfect, layout of coordinates.

**5. Physics Simulation (Settlement)**
* **Trigger:** The repaired solution is ready for final verification.
* **Stage Process:** The solution is sent to `physics_settle()` (`optimizer_physics.py`). PyBullet spawns the items into a headless 3D physics environment. It simulates dropping the items with actual mass and friction to perfectly resolve micro-overlaps and gravity-related sliding over 1000 discrete simulation steps.
* **Output:** The physically settled, final resting coordinates `(x, y, z)` for each item.

**6. Fitness Evaluation**
* **Trigger:** The physical settlement is complete.
* **Stage Process:** The new physically settled coordinates are evaluated by `fitness_function_numpy()` (`optimizer.py`). The metrics for Space Utilization, Accessibility, and Stability are computed using the mathematical formulas outlined above.
* **Output:** A set of final fitness scores indicating the quality of the packing layout.

**7. Save & Render**
* **Trigger:** The fitness scores are calculated.
* **Stage Process:** `finalize_optimization()` (`app.py`) updates the item positions and saves the final metrics to `database.py`. The frontend (`script.js`), which was polling the server for status updates, detects the process completion, re-fetches the items, and neatly renders the finalized 3D layout in the browser.
* **Output:** The user sees the fully optimized warehouse packing layout rendered in the 3D canvas alongside the performance metrics.
