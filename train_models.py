import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
import glob
from ml_utils import PackingModel

import json
import time

# Configuration
DATA_DIR = "training_data"
MODELS_DIR = "models"
EPOCHS = 100               # Researcher-aligned epoch count for stable convergence
BATCH_SIZE = 2048          # Optimized for high-throughput GPU utilization
LR = 0.001
VAL_SPLIT = 0.2
PATIENCE = 10              # Improved early-stopping for faster convergence
HISTORY_LOG_DIR = os.path.join("Documents", "04_Machine_Learning", "Performance_Metrics")
os.makedirs(HISTORY_LOG_DIR, exist_ok=True)

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

class WarehouseDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
        # Original 10 features
        orig_x = self.data[['item_l', 'item_w', 'item_h', 'weight', 'fragile', 'stackable', 'can_rotate', 'wh_l', 'wh_w', 'wh_h']].values.astype(np.float32)
        
        # Derived features (8 more)
        l, w, h = orig_x[:, 0], orig_x[:, 1], orig_x[:, 2]
        wh_l, wh_w, wh_h = orig_x[:, 7], orig_x[:, 8], orig_x[:, 9]
        
        item_vol = l * w * h
        wh_vol = wh_l * wh_w * wh_h
        item_area = l * w
        wh_area = wh_l * wh_w
        
        # Build 19-feature set
        n = len(self.data)
        self.x = np.zeros((n, 19), dtype=np.float32)
        
        # 1-10: Basic
        self.x[:, 0:3] = orig_x[:, 0:3] / 10.0
        self.x[:, 3] = orig_x[:, 3] / 100.0
        self.x[:, 4:7] = orig_x[:, 4:7]
        self.x[:, 7:10] = orig_x[:, 7:10] / 100.0
        
        # 11-18: Advanced
        self.x[:, 10] = item_vol / 10.0
        self.x[:, 11] = wh_vol / 1000.0
        self.x[:, 12] = item_vol / (wh_vol + 1e-6)
        self.x[:, 13] = item_area / 10.0
        self.x[:, 14] = wh_area / 100.0
        self.x[:, 15] = item_area / (wh_area + 1e-6)
        self.x[:, 16] = l / (wh_l + 1e-6)
        self.x[:, 17] = w / (wh_w + 1e-6)
        
        # 19: Sequence Progress (Normalized index in dataset / CSV)
        # Note: In real scenarios this is item_index / total_items
        self.x[:, 18] = np.arange(n) / float(n)
        
        # Targets: x, y, z, rot
        self.y = self.data[['target_x', 'target_y', 'target_z', 'target_rot']].values.astype(np.float32)
        
        # Normalise targets by warehouse dims
        wh_l = self.data['wh_l'].values.astype(np.float32) + 1e-5
        wh_w = self.data['wh_w'].values.astype(np.float32) + 1e-5
        wh_h = self.data['wh_h'].values.astype(np.float32) + 1e-5
        
        self.y[:, 0] = self.y[:, 0] / wh_l
        self.y[:, 1] = self.y[:, 1] / wh_w
        self.y[:, 2] = self.y[:, 2] / wh_h
        self.y[:, 3] = self.y[:, 3] / 6.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


def train_model(csv_path, model_name):
    print(f"\nTraining model for {csv_path}...")
    dataset = WarehouseDataset(csv_path)

    # Train / Validation split
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Samples: {n_train} train, {n_val} val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PackingModel()
    model.to(device)
    
    # Weighted Loss: emphasize Z only for stacking-height accuracy
    # [x, y, z, rot] -> z has higher weight
    weight_v = torch.tensor([1.0, 1.0, 2.0, 1.0]).to(device)
    def weighted_mse_loss(input, target):
        return (weight_v * (input - target) ** 2).mean()

    def calculate_collision_penalty(pred, batch_x):
        """
        Penalizes overlapping bounding boxes within items of the same sequence (bin).
        Uses warehouse dimensions (features 7-9) to identify items in the same sequence.
        """
        # 1. Extract normalized sizes: L' = L/WH_L, W' = W/WH_W, H' = H/WH_H
        # Item dims (normalized by 10): indices 0,1,2
        # WH dims (normalized by 100): indices 7,8,9
        l_prime = (batch_x[:, 0] * 10.0) / (batch_x[:, 7] * 100.0 + 1e-6)
        w_prime = (batch_x[:, 1] * 10.0) / (batch_x[:, 8] * 100.0 + 1e-6)
        h_prime = (batch_x[:, 2] * 10.0) / (batch_x[:, 9] * 100.0 + 1e-6)
        
        # 2. Define Bounding Boxes in normalized unit space [0, 1]^3
        x1, y1, z1 = pred[:, 0], pred[:, 1], pred[:, 2]
        x2, y2, z2 = x1 + l_prime, y1 + w_prime, z1 + h_prime
        
        # 3. Pairwise Intersections (N x N) via broadcasting
        ix1 = torch.max(x1.unsqueeze(1), x1.unsqueeze(0))
        ix2 = torch.min(x2.unsqueeze(1), x2.unsqueeze(0))
        inter_x = torch.clamp(ix2 - ix1, min=0)
        
        iy1 = torch.max(y1.unsqueeze(1), y1.unsqueeze(0))
        iy2 = torch.min(y2.unsqueeze(1), y2.unsqueeze(0))
        inter_y = torch.clamp(iy2 - iy1, min=0)
        
        iz1 = torch.max(z1.unsqueeze(1), z1.unsqueeze(0))
        iz2 = torch.min(z2.unsqueeze(1), z2.unsqueeze(0))
        inter_z = torch.clamp(iz2 - iz1, min=0)
        
        overlap_vol = inter_x * inter_y * inter_z
        
        # 4. Mask: Only penalize if items are in the same sequence (share WH dims)
        # and ignore self-overlap (diagonal)
        wh_dims = batch_x[:, 7:10]
        same_seq = torch.all(wh_dims.unsqueeze(1) == wh_dims.unsqueeze(0), dim=2).float()
        mask = (1.0 - torch.eye(pred.size(0), device=device)) * same_seq
        
        return (overlap_vol * mask).sum() / (mask.sum() + 1e-6)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_mse": [], "train_coll": [], "lr": []}

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        total_loss = 0
        total_mse = 0
        total_coll = 0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            mse_loss = weighted_mse_loss(outputs, batch_y)
            coll_penalty = calculate_collision_penalty(outputs, batch_x)
            
            # Physics-Informed Loss: combine MSE with collision penalty
            loss = mse_loss + (10.0 * coll_penalty)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_coll += coll_penalty.item()
            n_batches += 1

        avg_train = total_loss / max(n_batches, 1)
        avg_mse = total_mse / max(n_batches, 1)
        avg_coll = total_coll / max(n_batches, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                val_loss += weighted_mse_loss(pred, batch_y).item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)
        scheduler.step()
        
        # Log history
        history["train_loss"].append(avg_train)
        history["train_mse"].append(avg_mse)
        history["train_coll"].append(avg_coll)
        history["val_loss"].append(avg_val)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            # Save best model
            save_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 1 == 0:
            print(f"  Epoch [{epoch+1}/{EPOCHS}] | Total: {avg_train:.6f} | MSE: {avg_mse:.6f} | Coll: {avg_coll:.6f} | Val: {avg_val:.6f} (Best: {best_val_loss:.6f})")

        if patience_counter >= PATIENCE:
            print(f"  [STOP] Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

    print(f"  [OK] Best val loss: {best_val_loss:.6f} -- saved to models/{model_name}.pth")
    return history, best_val_loss

def run_training():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("No CSV files found in training_data/")
        return

    all_histories = {}
    
    for csv_file in sorted(csv_files):
        basename = os.path.splitext(os.path.basename(csv_file))[0]
        model_name = f"model_{basename}"
        print(f"\n--- Training {model_name} ---")
        history, best_loss = train_model(csv_file, model_name)
        all_histories[model_name] = {
            "history": history,
            "best_val_loss": best_loss,
            "params": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "patience": PATIENCE
            }
        }

    # Save all histories to one JSON for visualization
    history_file = os.path.join(HISTORY_LOG_DIR, "ml_training_history.json")
    with open(history_file, 'w') as f:
        json.dump(all_histories, f, indent=4)
    print(f"\nAll training histories saved to {history_file}")

if __name__ == "__main__":
    run_training()
