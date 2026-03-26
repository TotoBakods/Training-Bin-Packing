import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
import glob
from ml_utils import PackingModel

# Configuration
DATA_DIR = "training_data"
MODELS_DIR = "models"
EPOCHS = 100
BATCH_SIZE = 128
LR = 0.001
VAL_SPLIT = 0.2
PATIENCE = 15          # Early-stopping patience

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
        
        # Build 18-feature set
        n = len(self.data)
        self.x = np.zeros((n, 18), dtype=np.float32)
        
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

    model = PackingModel()
    
    # Weighted Loss: 2.0x for X/Y coordinates to reduce displacement gap
    weight_v = torch.tensor([2.0, 2.0, 1.0, 1.0]).to(device if 'device' in locals() else 'cpu')
    def weighted_mse_loss(input, target):
        return (weight_v * (input - target) ** 2).mean()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        total_loss = 0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = weighted_mse_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1

        avg_train = total_loss / max(n_batches, 1)

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

        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            # Save best model
            save_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{EPOCHS}], Train: {avg_train:.6f}, Val: {avg_val:.6f} (best: {best_val_loss:.6f}, lr: {optimizer.param_groups[0]['lr']:.2e})")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

    print(f"  [OK] Best val loss: {best_val_loss:.6f} -- saved to models/{model_name}.pth")

def run_training():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("No CSV files found in training_data/")
        return

    for csv_file in sorted(csv_files):
        basename = os.path.splitext(os.path.basename(csv_file))[0]
        train_model(csv_file, f"model_{basename}")

if __name__ == "__main__":
    run_training()
