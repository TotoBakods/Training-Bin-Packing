# Bin Packing Optimization System - Core Use Cases

This document outlines the core system use cases and interactions for the Bin Packing Optimization System, focusing on the high-level flow of the application.

## Use Case Diagram

```mermaid
flowchart LR
    %% Custom Styles
    classDef actor fill:#2c3e50,stroke:#1a252f,stroke-width:2px,color:#fff,font-weight:bold
    classDef usecase fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#0d47a1,font-weight:bold,rx:15,ry:15
    classDef sysBoundary fill:#fafafa,stroke:#90a4ae,stroke-width:3px,stroke-dasharray: 5 5,color:#263238,font-weight:bold

    %% Actors
    User(("🧑‍💼 User / Warehouse Manager")):::actor
    System(("💻 System (Flask API)")):::actor
    Optimizer(("🧠 Optimizer (ML/GA/EO)")):::actor

    subgraph SystemBoundary ["Core Operations"]
        direction TB

        UC1(["Prepare Data (Import/Manage)"]):::usecase
        UC2(["Configure Packing Logic (Weights)"]):::usecase
        UC3(["Execute Bin Packing Optimization"]):::usecase
        UC4(["Validate & Settle Physically"]):::usecase
        UC5(["Review 3D Packing Results"]):::usecase
        UC6(["Export Final Manifest"]):::usecase
    end

    %% Apply Classes
    class SystemBoundary sysBoundary

    %% User Interactions (Start & End points)
    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC5
    User --> UC6

    %% System Engine Interactions
    UC3 <--- System
    UC3 <--- Optimizer
    UC4 <--- System
    
    %% Internal Use Case Links (Sequence)
    UC1 -. "provides data to" .-> UC3
    UC2 -. "configures" .-> UC3
    UC3 -. "<< include >>" .-> UC4
    UC4 -. "generates view for" .-> UC5
```

## Actor Descriptions

1. **User / Warehouse Manager**: Responsible for supplying warehouse and item data, starting optimization jobs, and reviewing final packing plans.
2. **System (Flask API)**: Acts as the middleman to handle the HTTP requests, interact with the sqlite database, coordinate the optimization threads, and apply physical simulations.
3. **Optimizer (ML/GA/EO)**: The computational engine responsible for calculating the multi-objective fitness metrics and exploring spatial layouts for maximum efficiency.

## Core System Operations

* **Prepare Data (Import/Manage)**: The user manages the layout of the warehouse and populates the items to pack, taking advantage of automated CSV importing or generation capabilities provided by the backend.
* **Configure Packing Logic**: Setting the parameters (e.g. generations or iterations limits) and deciding the fitness weights favoring space, accessibility, or stability prior to running the model.
* **Execute Bin Packing Optimization**: The core process where the system bridges the active warehouse configuration to the Optimizer thread, running models over many generations to discover the best subset of geometric item coordinates.
* **Validate & Settle Physically**: Ensures the selected solution passes the Pybullet gravity rules, removing clipping conflicts and securing items based on physics rules before locking the layout into the database.
* **Review 3D Packing Results**: The system outputs a visual mapping utilizing `three.js`, giving the user a 3D perspective on their finalized packing plan.
* **Export Final Manifest**: After validation and review, the user can download a serialized CSV sequence directing physical warehouse workers exactly where to place each parcel for maximum calculated efficiency.
