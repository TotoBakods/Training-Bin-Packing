import time
import numpy as np
from ml_utils import MLOptimizer
from database import get_all_items, get_warehouse_config

def test_speed(variant):
    print(f"\nTesting {variant}...")
    items = get_all_items(1)
    warehouse = get_warehouse_config(1)
    if not items:
        print("No items found. Load sample data first.")
        return
    
    optimizer = MLOptimizer(variant)
    start_time = time.time()
    solution, fitness, ttb, metrics = optimizer.optimize(items, warehouse)
    end_time = time.time()
    
    print(f"Total Time: {end_time - start_time:.2f}s")
    print(f"Repair Latency: {metrics.get('repair_latency_ms', 0):.2f}ms")
    print(f"Fitness: {fitness:.4f}")
    print(f"Placed Count: {metrics.get('placed_count', 0)}")

if __name__ == "__main__":
    # Test GA-EO (Normal Mode)
    test_speed("fit_ga_eo")
    # Test EO-GA (Fast Mode - should be much faster)
    test_speed("fit_eo_ga")
