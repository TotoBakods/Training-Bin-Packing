# Bin Packing Optimization System - Use Case Diagram

This document contains a simplified use case diagram and descriptions for the Bin Packing Optimization System.

## Use Case Diagram

```mermaid
flowchart LR
    %% Custom Styles
    classDef actor fill:#2c3e50,stroke:#1a252f,stroke-width:2px,color:#fff,font-weight:bold
    classDef usecase fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#0d47a1,font-weight:bold,rx:15,ry:15
    classDef sysBoundary fill:#fafafa,stroke:#90a4ae,stroke-width:3px,stroke-dasharray: 5 5,color:#263238,font-weight:bold

    %% Actors
    User(("🧑‍💼 Warehouse Manager")):::actor
    Optimizer(("🧠 ML Optimizer Engine")):::actor
    Physics(("⚙️ Physics Engine")):::actor

    subgraph SystemBoundary ["Bin Packing Optimization System"]
        direction TB

        UC1(["Manage Warehouse Configuration"]):::usecase
        UC2(["Manage Inventory & Constraints"]):::usecase
        UC3(["Import/Export Data (CSV/Manifest)"]):::usecase
        
        UC4(["Run Optimization (GA/EO/Hybrid)"]):::usecase
        UC5(["Calculate Fitness Metrics"]):::usecase
        
        UC6(["Simulate Physics (PyBullet)"]):::usecase
        
        UC7(["View Results & 3D Visualization"]):::usecase
    end

    %% Apply Classes
    class SystemBoundary sysBoundary

    %% User Interactions
    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC7

    %% System Engine Interactions (connecting from right)
    UC5 <--- Optimizer
    UC4 <--- Optimizer
    
    UC6 <--- Physics

    %% Internal Use Case Links (includes & extends)
    UC4 -. "<< include >>" .-> UC5
    
    UC5 -. "<< extend >>" .-> UC6
```

## Actor Descriptions

1. **Warehouse Manager / User**: The primary user who manages layouts, imports items, configures optimization, and views results.
2. **ML Optimizer Engine**: The backend module that uses algorithmic models (GA, EO, Hybrid) to calculate spatial packing arrangements.
3. **Physics Engine**: The physical simulation (PyBullet) that mathematically settles the placed objects.

## Use Case Descriptions

* **Manage Warehouse Configuration**: Allows users to create, view, switch, and update the geometric constraints and layouts of warehouses.
* **Manage Inventory & Constraints**: Allows users to add, edit, or delete items within a warehouse, and define physical exclusion zones where items cannot be packed.
* **Import/Export Data**: Facilitates the bulk uploading of `.csv` inventories, loading of sample data, and the export of completed packing manifests.
* **Run Optimization**: Triggers the machine learning heuristic algorithms (Genetic Algorithm, Extremal Optimization, or Hybrid models) to generate packing solutions.
* **Calculate Fitness Metrics**: Evaluates a packing solution according to its space utilization, grouping, accessibility, and stability attributes. Included as part of the optimization process.
* **Simulate Physics (PyBullet)**: An optional extension of the fitness calculation that resolves clipping and settles items gravitationally.
* **View Results & 3D Visualization**: Provides the user with an interactive 3D WebGL rendering of the packing configuration alongside performance analytics.
