"""
Generate Training Data for ML Packing Models (v2)
==================================================
Uses the GAN to create realistic items, then runs the repair_solution_compact
heuristic to produce physically-valid (x, y, z, rot) targets.

v2 improvements:
  - 10,000+ rows per heuristic variant (200 scenarios x 50 items).
  - Mix of DENSE + NORMAL warehouse scenarios to force Z stacking.
  - Dense scenarios use small floor areas + many items so the heuristic
    MUST stack items vertically (Z > 0).
  - Normal scenarios maintain variety for x/y position learning.
"""

import os
import sys
import random
import uuid

import numpy as np
import pandas as pd
import torch
import pickle
import json
import time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR   = os.path.join(SCRIPT_DIR, "training_data")
MODELS_DIR     = os.path.join(SCRIPT_DIR, "models")
GAN_DIR        = os.path.join(SCRIPT_DIR, "gan")
BATCH_SIZE = 1024   # Optimized for high-throughput VRAM training
EPOCHS = 80         # Valid professional standard epoch count
VAL_SPLIT = 0.20
LR = 0.001
PATIENCE = 15

# GAN assets
SCALER_PATH = os.path.join(GAN_DIR, "scaler.pkl")
GENERATOR_PATH = os.path.join(GAN_DIR, "checkpoints", "generator.pth")

# Make sure output dirs exist
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, GAN_DIR)

from gan.model import Generator
from optimizer import repair_solution_compact, get_rotated_dims

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LATENT_DIM = 100

# High-Throughput Scenarios for Quick Research Validation (Reduced temporarily)
DENSE_SCENARIOS  = 120  # 60% Dense
NORMAL_SCENARIOS = 80   # 40% Standard scenarios
TOTAL_SCENARIOS  = DENSE_SCENARIOS + NORMAL_SCENARIOS
ITEMS_PER_SCENARIO = 50

# We use one primary dataset for all models to ensure scientific parity
TRAINING_FILENAME = "warehouse_training.csv"

# Category heuristics (same as gan/generate.py)
FRAGILE_CATEGORIES = {
    'confectionery', 'bakery products', 'fruit', 'vegetables',
    'ice cream', 'eggs', 'glass', 'electronics'
}


def load_gan():
    """Load trained GAN generator + scaler."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"GAN scaler not found at {SCALER_PATH}. Run gan/train.py first.")
    if not os.path.exists(GENERATOR_PATH):
        raise FileNotFoundError(f"GAN generator not found at {GENERATOR_PATH}. Run gan/train.py first.")

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    generator = Generator(LATENT_DIM, 4).to(device)
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=device, weights_only=True))
    generator.eval()

    return generator, scaler, device


def generate_items(generator, scaler, device, n_items, category_pool=None):
    """Generate n synthetic items via GAN."""
    z = torch.randn(n_items, LATENT_DIM).to(device)
    with torch.no_grad():
        gen_data = generator(z).cpu().numpy()

    original = scaler.inverse_transform(gen_data)

    if category_pool is None:
        category_pool = ["General", "Electronics", "Clothing", "Furniture", "Books"]

    items = []
    for i in range(n_items):
        l, w, h, weight = original[i]
        # Use real GAN output dimensions (removed legacy 2.0x multiplier)
        l, w, h = abs(l), abs(w), abs(h)
        weight = max(abs(weight), 0.1)

        cat = random.choice(category_pool)
        is_fragile = 1 if cat.lower() in FRAGILE_CATEGORIES else 0
        is_stackable = 0 if is_fragile else (1 if random.random() > 0.05 else 0)
        can_rotate = 0 if h > 2 * min(l, w) else 1

        items.append({
            "length": round(float(l), 2),
            "width":  round(float(w), 2),
            "height": round(float(h), 2),
            "weight": round(float(weight), 2),
            "fragile": is_fragile,
            "stackable": is_stackable,
            "can_rotate": can_rotate,
            "access_freq": random.randint(1, 10),
            "category": cat,
        })

    return items


def items_to_props(items):
    """Convert item dicts to items_props array (N, 9)."""
    n = len(items)
    props = np.zeros((n, 9), dtype=np.float32)
    for i, item in enumerate(items):
        props[i] = [
            item["length"], item["width"], item["height"],
            item["can_rotate"], item["stackable"],
            item["access_freq"], item["weight"],
            hash(item.get("category", "")) % 10000,
            item.get("fragile", 0),
        ]
    return props


def dense_warehouse(items):
    """
    Generate a warehouse whose floor area is SMALLER than the total
    item footprint so that items are forced to stack vertically.
    
    Strategy: calculate total item footprint, then make the warehouse
    floor ~30-70% of that footprint. This guarantees multi-layer stacking.
    """
    total_footprint = sum(item["length"] * item["width"] for item in items)
    
    # Target floor area = 30-70% of total item footprint (forces 2-3 layers)
    floor_ratio = random.uniform(0.3, 0.7)
    target_area = total_footprint * floor_ratio
    
    # Random aspect ratio
    aspect = random.uniform(0.6, 1.6)
    wh_l = round(max(2.0, (target_area * aspect) ** 0.5), 1)
    wh_w = round(max(2.0, target_area / wh_l), 1)
    wh_h = round(random.uniform(5.0, 12.0), 1)
    
    return wh_l, wh_w, wh_h


def normal_warehouse():
    """Generate a normal-sized warehouse for x/y diversity."""
    wh_l = round(random.uniform(8.0, 25.0), 1)
    wh_w = round(random.uniform(6.0, 20.0), 1)
    wh_h = round(random.uniform(5.0, 12.0), 1)
    return wh_l, wh_w, wh_h


def create_initial_solution(n, wh_l, wh_w, wh_h):
    """Create a random initial solution."""
    solution = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        solution[i, 0] = random.uniform(0, wh_l)
        solution[i, 1] = random.uniform(0, wh_w)
        solution[i, 2] = 0.0
        solution[i, 3] = random.choice([0, 1])
    return solution


def generate_dataset_for_variant(variant_name, generator, scaler, device):
    """Generate one full CSV of training data for a heuristic variant."""
    all_rows = []
    z_positive_count = 0
    total_count = 0

    total_scenarios = DENSE_SCENARIOS + NORMAL_SCENARIOS
    print(f"\n  Generating data for '{variant_name}' ...")
    print(f"    {DENSE_SCENARIOS} dense + {NORMAL_SCENARIOS} normal = {total_scenarios} scenarios")

    for sample_idx in range(total_scenarios):
        is_dense = sample_idx < DENSE_SCENARIOS

        # Generate items first (need them for dense warehouse sizing)
        n_items = ITEMS_PER_SCENARIO
        items = generate_items(generator, scaler, device, n_items)
        items_props = items_to_props(items)

        # Choose warehouse size
        if is_dense:
            wh_l, wh_w, wh_h = dense_warehouse(items)
        else:
            wh_l, wh_w, wh_h = normal_warehouse()

        # Create initial random solution
        solution = create_initial_solution(n_items, wh_l, wh_w, wh_h)

        # Run repair heuristic to get physically-valid positions
        wh_dims = (wh_l, wh_w, wh_h, 0, 0)
        valid_z = [0.0]

        repaired = repair_solution_compact(
            solution.copy(), items_props, wh_dims,
            allocation_zones=None, layer_heights=valid_z
        )

        # Build rows
        for i in range(n_items):
            item = items[i]
            z_val = float(repaired[i, 2])
            if z_val > 0.01:
                z_positive_count += 1
            total_count += 1

            row = {
                "item_l":     item["length"],
                "item_w":     item["width"],
                "item_h":     item["height"],
                "weight":     item["weight"],
                "fragile":    item["fragile"],
                "stackable":  item["stackable"],
                "can_rotate": item["can_rotate"],
                "wh_l":       wh_l,
                "wh_w":       wh_w,
                "wh_h":       wh_h,
                "target_x":   float(repaired[i, 0]),
                "target_y":   float(repaired[i, 1]),
                "target_z":   z_val,
                "target_rot": int(repaired[i, 3]),
            }
            all_rows.append(row)

        if (sample_idx + 1) % 20 == 0:
            pct_z = z_positive_count / max(total_count, 1) * 100
            print(f"    ... {sample_idx + 1}/{total_scenarios} scenarios done  "
                  f"({len(all_rows)} rows, {pct_z:.1f}% with Z>0)")

    # Write CSV
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(TRAINING_DIR, f"{variant_name}.csv")
    df.to_csv(out_path, index=False)
    
    # Log Summary
    pct_z = z_positive_count / max(total_count, 1) * 100
    summary = {
        "variant": variant_name,
        "rows": len(df),
        "stacking_pct": round(pct_z, 2),
        "avg_l": round(df["item_l"].mean(), 3),
        "avg_w": round(df["item_w"].mean(), 3),
        "avg_h": round(df["item_h"].mean(), 3),
        "dense_scenarios": DENSE_SCENARIOS,
        "normal_scenarios": NORMAL_SCENARIOS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    summary_path = os.path.join(TRAINING_DIR, f"{variant_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"  [OK] Saved {len(df)} rows to {out_path}  ({pct_z:.1f}% with Z>0)")
    return df


def main():
    total_scenarios = DENSE_SCENARIOS + NORMAL_SCENARIOS
    print("=" * 60)
    print("  Training Data Generation (v2 - Dense + Normal)")
    print("=" * 60)
    print(f"  Dense scenarios       : {DENSE_SCENARIOS}")
    print(f"  Normal scenarios      : {NORMAL_SCENARIOS}")
    print(f"  Total scenarios/variant: {total_scenarios}")
    print(f"  Items per scenario    : {ITEMS_PER_SCENARIO}")
    print(f"  Expected rows / CSV   : ~{total_scenarios * ITEMS_PER_SCENARIO}")
    print(f"  Target File           : {TRAINING_FILENAME}")
    print()

    # Load GAN
    print("[1/2] Loading GAN generator ...")
    generator, scaler, device = load_gan()
    print(f"  GAN loaded on {device}\n")

    # Generate the single master dataset
    print("[2/2] Generating Master Dataset ...")
    generate_dataset_for_variant("warehouse_training", generator, scaler, device)

    print("\n" + "=" * 60)
    print("  All datasets generated! Next steps:")
    print("    1. python train_models.py")
    print("    2. python evaluate_metrics.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
