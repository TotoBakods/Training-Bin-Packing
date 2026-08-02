import time
import numpy as np
from ml_utils import MLOptimizer
from database import get_all_items, get_warehouse_config

def test_speed():
    items = get_all_items(1)
    warehouse = get_warehouse_config(1)
    if not items:
        return
    
    print(f"Num Items: {len(items)}")
    
    results = {}
    for variant in ["fit_ga_eo", "fit_eo_ga"]:
        optimizer = MLOptimizer(variant)
        start = time.time()
        _, fitness, _, metrics = optimizer.optimize(items, warehouse)
        results[variant] = {
            "total_time": time.time() - start,
            "repair_ms": metrics.get('repair_latency_ms', 0),
            "fitness": fitness
        }
    
    for v, r in results.items():
        print(f"{v}: {r['total_time']:.2f}s (Repair: {r['repair_ms']:.2f}ms), Fitness: {r['fitness']:.4f}")

if __name__ == "__main__":
    test_speed()
