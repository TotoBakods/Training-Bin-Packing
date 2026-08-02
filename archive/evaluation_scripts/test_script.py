import sys
import numpy as np
sys.path.append('c:/Users/jebzw/OneDrive/Documents/Training-Bin-Packing')

from optimizer import repair_solution_compact
from ml_utils import MLOptimizer
from database import get_warehouse_config, get_all_items

warehouse = get_warehouse_config(1)
items = get_all_items(1)
print(f"Total items: {len(items)}")

optimizer = MLOptimizer('fit_eo_ga')
best_solution, best_fitness, time_to_best = optimizer.optimize(items, warehouse)
print(f"Length of solution: {len(best_solution)}")

placed = 0
unplaced = 0
for s in best_solution:
    if s['z'] < 1000:
        placed += 1
    else:
        unplaced += 1
        print(f"Unplaced item ID {s['id']} at Z = {s['z']}")
print(f"Placed: {placed}, Unplaced: {unplaced}")
