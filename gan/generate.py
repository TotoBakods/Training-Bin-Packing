import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import argparse
import os
import uuid
import random
try:
    from .model import Generator
except ImportError:
    from model import Generator

# Parameters
LATENT_DIM = 100
current_dir = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(current_dir, 'checkpoints', 'generator.pth')
SCALER_PATH = os.path.join(current_dir, 'scaler.pkl')
DEFAULT_OUTPUT_FILE = os.path.join(current_dir, 'generated_items.csv')
REAL_DATA_FILE = os.path.join(current_dir, '..', 'datasets', 'datasets.csv')

# Heuristics
FRAGILE_CATEGORIES = {
    'confectionery', 'bakery products', 'fruit', 'vegetables', 
    'ice cream', 'eggs', 'glass', 'electronics'
}

def get_category_distribution():
    try:
        df = pd.read_csv(REAL_DATA_FILE)
        return df['category'].value_counts(normalize=True).to_dict()
    except Exception as e:
        print(f"Warning: Could not load categories from {REAL_DATA_FILE}: {e}")
        return {'General': 1.0}

def generate(n_items=100, warehouse_length=20.0, warehouse_width=15.0, scale_factor=2.0, seed=None, output_file=None):
    """
    Generate synthetic items using the trained GAN.
    
    Args:
        n_items: Number of items to generate
        warehouse_length: Warehouse length for position calculation
        warehouse_width: Warehouse width for position calculation  
        scale_factor: Multiplier for item dimensions (default 2.0 for larger items)
        seed: Random seed for reproducibility
        output_file: Path to save the generated CSV
    """
    # Guard: Ensure GAN training is finished
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: GAN checkpoint not found at {CHECKPOINT_PATH}. Aborting generator.")
        return
    
    # Check if history exists and reached 500 epochs
    history_path = os.path.join(current_dir, 'loss_history.json')
    if os.path.exists(history_path):
        import json
        with open(history_path, 'r') as f:
            hist = json.load(f)
            if hist.get('epochs', 0) < 500:
                print(f"Warning: GAN training only reached {hist.get('epochs')} epochs. Pipeline requires 500. Aborting.")
                return

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Using random seed: {seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Scaler
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Run train.py first.")
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    # Load Model
    model = Generator(LATENT_DIM, 4).to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}. Run train.py first.")
        
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    
    # Get Category Dist
    cat_dist = get_category_distribution()
    categories = list(cat_dist.keys())
    probs = list(cat_dist.values())
    
    # Generate
    z = torch.randn(n_items, LATENT_DIM).to(device)
    with torch.no_grad():
        generated_data = model(z).cpu().numpy()
        
    # Inverse Transform
    original_scale_data = scaler.inverse_transform(generated_data)
    
    items = []
    for i in range(n_items):
        l, w, h, weight = original_scale_data[i]
        
        # Ensure positive and apply scale factor for larger items
        l, w, h = abs(l) * scale_factor, abs(w) * scale_factor, abs(h) * scale_factor
        weight = abs(weight) * scale_factor  # Scale weight proportionally
        
        # Assign Category
        category = random.choices(categories, weights=probs, k=1)[0]
        
        # Logic: Fragility & Stackable
        is_fragile = 1 if category in FRAGILE_CATEGORIES else 0
        # If fragile, not stackable. If not fragile, 90% chance stackable
        is_stackable = 0 if is_fragile else (1 if random.random() > 0.1 else 0)
        
        # Logic: Priority & Access
        priority = random.randint(1, 3)
        access_freq = random.randint(1, 10)
        
        # Logic: Rotation
        # Unstable items (very tall) shouldn't rotate
        can_rotate = 0 if h > 2 * min(l, w) else 1
        
        # Generate random position within warehouse bounds
        # Ensure item fits within warehouse (item center position)
        min_x = l / 2
        max_x = warehouse_length - l / 2
        min_y = w / 2
        max_y = warehouse_width - w / 2
        
        # Clamp to valid range
        if max_x < min_x:
            max_x = min_x
        if max_y < min_y:
            max_y = min_y
        
        x = round(random.uniform(min_x, max_x), 2)
        y = round(random.uniform(min_y, max_y), 2)
        z_pos = 0.0  # Start on floor
        rotation = random.choice([0, 90, 180, 270]) if can_rotate else 0

        items.append({
            'id': f'SYN-{uuid.uuid4().hex[:8]}',
            'name': f'Synthetic {category} {i+1}',
            'length': round(float(l), 2),
            'width': round(float(w), 2),
            'height': round(float(h), 2),
            'weight': round(float(weight), 2),
            'category': category,
            'priority': priority,
            'fragility': is_fragile,
            'stackable': is_stackable,
            'access_freq': access_freq,
            'can_rotate': can_rotate,
            #'x': x,
            #'y': y,
            #'z': z_pos,
            #'rotation': rotation
        })
        
    df = pd.DataFrame(items)
    
    # Reorder columns to include standard item props
    cols = ['id', 'name', 'length', 'width', 'height', 'weight', 'category', 
            'priority', 'fragility', 'stackable', 'access_freq', 'can_rotate']
    df = df[cols]
    
    save_path = output_file if output_file else DEFAULT_OUTPUT_FILE
    df.to_csv(save_path, index=False)
    print(f"Generated {n_items} items with positions and saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000, help='Number of items to generate')
    parser.add_argument('--output', type=str, default=None, help='Output filename or absolute path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # If output is just a filename, put it in the gan directory
    out_path = args.output
    if out_path and not os.path.isabs(out_path):
        out_path = os.path.join(current_dir, out_path)

    generate(args.n, output_file=out_path, seed=args.seed)
