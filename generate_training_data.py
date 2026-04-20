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

# Scenario counts — 3 types to capture all placement behaviours
DENSE_SCENARIOS    = 120   # 40% Dense (forces vertical stacking)
NORMAL_SCENARIOS   = 100   # 33% Normal variety
DOOR_AF_SCENARIOS  = 80    # 27% High-AF door-cluster (trains door-proximity behaviour)
TOTAL_SCENARIOS    = DENSE_SCENARIOS + NORMAL_SCENARIOS + DOOR_AF_SCENARIOS
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


def generate_items(generator, scaler, device, n_items, category_pool=None,
                   high_af_bias=False, fragile_floor=0.25):
    """
    Generate n synthetic items via GAN.

    Parameters
    ----------
    high_af_bias   : bool  – when True, skews access_freq toward 5–10 so the
                             model sees more door-cluster examples.
    fragile_floor  : float – minimum fraction of items that must be fragile.
                             Ensures upper-zone training is well represented.
    """
    z = torch.randn(n_items, LATENT_DIM).to(device)
    with torch.no_grad():
        gen_data = generator(z).cpu().numpy()

    original = scaler.inverse_transform(gen_data)

    if category_pool is None:
        category_pool = ["General", "Electronics", "Clothing", "Furniture", "Books"]

    # Fragile categories pool for forced fragile items
    fragile_pool = list(FRAGILE_CATEGORIES)

    items = []
    n_forced_fragile = max(1, int(n_items * fragile_floor))

    for i in range(n_items):
        l, w, h, weight = original[i]
        l, w, h = abs(l), abs(w), abs(h)
        weight = max(abs(weight), 0.1)

        # Force fragile_floor fraction to be fragile so upper-zone targets
        # appear in every scenario.
        if i < n_forced_fragile:
            cat = random.choice(fragile_pool)
        else:
            cat = random.choice(category_pool)

        is_fragile = 1 if cat.lower() in FRAGILE_CATEGORIES else 0
        is_stackable = 0 if is_fragile else (1 if random.random() > 0.05 else 0)
        can_rotate = 0 if h > 2 * min(l, w) else 1

        # High-AF bias: for door-cluster scenarios use randint(5,10) more often
        if high_af_bias:
            af = random.randint(5, 10) if random.random() < 0.65 else random.randint(1, 5)
        else:
            af = random.randint(1, 10)

        prio = random.randint(1, 3)

        items.append({
            "length":      round(float(l), 2),
            "width":       round(float(w), 2),
            "height":      round(float(h), 2),
            "weight":      round(float(weight), 2),
            "fragile":     is_fragile,
            "stackable":   is_stackable,
            "can_rotate":  can_rotate,
            "access_freq": af,
            # Priority 1=low, 2=medium, 3=high — determines wall-proximity preference
            "priority":    prio,
            # Composite key used by the optimizer — teach the model directly
            "af_prio":     af * prio,
            "category":    cat,
        })

    # Shuffle so forced-fragile items aren't always first in the placement order
    random.shuffle(items)
    return items


def items_to_props(items):
    """Convert item dicts to items_props array (N, 10) — includes priority."""
    n = len(items)
    props = np.zeros((n, 10), dtype=np.float32)
    for i, item in enumerate(items):
        props[i] = [
            item["length"], item["width"], item["height"],
            item["can_rotate"], item["stackable"],
            item["access_freq"], item["weight"],
            hash(item.get("category", "")) % 10000,
            item.get("fragile", 0),
            item.get("priority", 1),           # index 9
        ]
    return props


def _door_cluster_warehouse():
    """
    Small-to-medium warehouse with the door placed on the short wall (front/back)
    so that door-cluster scenarios have a well-defined door side.
    """
    wh_l = round(random.uniform(8.0, 20.0), 1)
    wh_w = round(random.uniform(5.0, 12.0), 1)
    wh_h = round(random.uniform(5.0, 10.0), 1)
    # Place door on front (y=0) or back (y=wh_w) wall, centred ±20%
    front = random.random() < 0.5
    door_x = round(wh_l * random.uniform(0.3, 0.7), 1)
    door_y = 0.0 if front else wh_w
    return wh_l, wh_w, wh_h, door_x, door_y


def _random_door(wh_l, wh_w):
    """Place the door randomly on one of the four warehouse walls."""
    wall = random.randint(0, 3)
    if wall == 0:   return round(random.uniform(0, wh_l), 1), 0.0          # front
    if wall == 1:   return round(random.uniform(0, wh_l), 1), wh_w         # back
    if wall == 2:   return 0.0, round(random.uniform(0, wh_w), 1)          # left
    return wh_l, round(random.uniform(0, wh_w), 1)                         # right


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
    door_x, door_y = _random_door(wh_l, wh_w)

    return wh_l, wh_w, wh_h, door_x, door_y


def normal_warehouse():
    """Generate a normal-sized warehouse for x/y diversity."""
    wh_l = round(random.uniform(8.0, 25.0), 1)
    wh_w = round(random.uniform(6.0, 20.0), 1)
    wh_h = round(random.uniform(5.0, 12.0), 1)
    door_x, door_y = _random_door(wh_l, wh_w)
    return wh_l, wh_w, wh_h, door_x, door_y


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
    """Generate one full CSV of training data for a heuristic variant.

    Three scenario types are interleaved:
      0 .. DENSE_SCENARIOS-1              → dense (forced stacking)
      DENSE_SCENARIOS .. +NORMAL-1        → normal variety
      DENSE+NORMAL .. +DOOR_AF-1          → high-AF door-cluster (door proximity)
    """
    all_rows = []
    z_positive_count = 0
    fragile_count = 0
    total_count = 0

    total_scenarios = DENSE_SCENARIOS + NORMAL_SCENARIOS + DOOR_AF_SCENARIOS
    print(f"\n  Generating data for '{variant_name}' ...")
    print(f"    {DENSE_SCENARIOS} dense + {NORMAL_SCENARIOS} normal "
          f"+ {DOOR_AF_SCENARIOS} door-AF = {total_scenarios} scenarios")

    for sample_idx in range(total_scenarios):
        is_dense   = sample_idx < DENSE_SCENARIOS
        is_door_af = sample_idx >= DENSE_SCENARIOS + NORMAL_SCENARIOS

        n_items = ITEMS_PER_SCENARIO

        # Generate items — door-AF scenarios skew access_freq higher and
        # keep fragile floor at 0.30 for all types.
        items = generate_items(
            generator, scaler, device, n_items,
            high_af_bias=is_door_af,
            fragile_floor=0.30,
        )
        items_props = items_to_props(items)

        # Choose warehouse geometry
        if is_dense:
            wh_l, wh_w, wh_h, door_x, door_y = dense_warehouse(items)
        elif is_door_af:
            wh_l, wh_w, wh_h, door_x, door_y = _door_cluster_warehouse()
        else:
            wh_l, wh_w, wh_h, door_x, door_y = normal_warehouse()

        # Create random initial solution then repair to get valid targets
        solution = create_initial_solution(n_items, wh_l, wh_w, wh_h)
        wh_dims = (wh_l, wh_w, wh_h, door_x, door_y)
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
            if item["fragile"]:
                fragile_count += 1
            total_count += 1

            row = {
                "scenario_id":  sample_idx,
                "item_l":       item["length"],
                "item_w":       item["width"],
                "item_h":       item["height"],
                "weight":       item["weight"],
                "fragile":      item["fragile"],
                "stackable":    item["stackable"],
                "can_rotate":   item["can_rotate"],
                "access_freq":  item["access_freq"],
                "priority":     item["priority"],
                # Composite ranking key used by the optimizer
                "af_prio":      item["af_prio"],
                "wh_l":         wh_l,
                "wh_w":         wh_w,
                "wh_h":         wh_h,
                "door_x":       door_x,
                "door_y":       door_y,
                "target_x":     float(repaired[i, 0]),
                "target_y":     float(repaired[i, 1]),
                "target_z":     z_val,
                "target_rot":   int(repaired[i, 3]),
            }
            all_rows.append(row)

        if (sample_idx + 1) % 20 == 0:
            pct_z    = z_positive_count / max(total_count, 1) * 100
            pct_frag = fragile_count     / max(total_count, 1) * 100
            print(f"    ... {sample_idx + 1}/{total_scenarios} scenarios done  "
                  f"({len(all_rows)} rows, Z>0={pct_z:.1f}%, fragile={pct_frag:.1f}%)")

    # Write CSV
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(TRAINING_DIR, f"{variant_name}.csv")
    df.to_csv(out_path, index=False)

    # Log summary
    pct_z    = z_positive_count / max(total_count, 1) * 100
    pct_frag = fragile_count     / max(total_count, 1) * 100
    summary = {
        "variant":          variant_name,
        "rows":             len(df),
        "stacking_pct":     round(pct_z, 2),
        "fragile_pct":      round(pct_frag, 2),
        "avg_l":            round(df["item_l"].mean(), 3),
        "avg_w":            round(df["item_w"].mean(), 3),
        "avg_h":            round(df["item_h"].mean(), 3),
        "dense_scenarios":  DENSE_SCENARIOS,
        "normal_scenarios": NORMAL_SCENARIOS,
        "door_af_scenarios": DOOR_AF_SCENARIOS,
        "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S")
    }
    summary_path = os.path.join(TRAINING_DIR, f"{variant_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"  [OK] Saved {len(df)} rows to {out_path}")
    print(f"       Z>0={pct_z:.1f}%  fragile={pct_frag:.1f}%")
    return df


def main():
    total_scenarios = DENSE_SCENARIOS + NORMAL_SCENARIOS + DOOR_AF_SCENARIOS
    print("=" * 60)
    print("  Training Data Generation (v3 - Dense + Normal + Door-AF)")
    print("=" * 60)
    print(f"  Dense scenarios       : {DENSE_SCENARIOS}")
    print(f"  Normal scenarios      : {NORMAL_SCENARIOS}")
    print(f"  Door-AF scenarios     : {DOOR_AF_SCENARIOS}")
    print(f"  Total scenarios       : {total_scenarios}")
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
