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
from logger_config import get_logger

logger = get_logger('training')

# ── GPU setup ────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    # cuDNN auto-tuner: finds the fastest conv algorithm for fixed input sizes
    torch.backends.cudnn.benchmark = True
    _gpu_name = torch.cuda.get_device_name(0)
    _vram_gb  = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    logger.info(f"[GPU] {_gpu_name}  {_vram_gb} GB VRAM  — training on CUDA")
else:
    logger.info("[GPU] No CUDA device found — falling back to CPU")

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR   = "training_data"
MODELS_DIR = "models"
EPOCHS      = 200
# 4096 fits comfortably in 12 GB VRAM for a 20-feature MLP; increase to 8192
# if VRAM usage stays below 8 GB during the first epoch.
BATCH_SIZE  = 4096
LR          = 0.001
VAL_SPLIT   = 0.2
PATIENCE    = 20
WARMUP_EPOCHS = 5
ITEMS_PER_SCENARIO = 50    # Matches generate_training_data.py
# Pin memory speeds up CPU→GPU transfer; only effective with CUDA
PIN_MEMORY  = torch.cuda.is_available()
# Workers for DataLoader — 0 is safest on Windows (avoids multiprocessing issues)
NUM_WORKERS = 0
HISTORY_LOG_DIR = os.path.join("Documents", "04_Machine_Learning", "Performance_Metrics")
os.makedirs(HISTORY_LOG_DIR, exist_ok=True)

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

class WarehouseDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        n = len(self.data)
        item_l  = self.data['item_l'].values.astype(np.float32)
        item_w  = self.data['item_w'].values.astype(np.float32)
        item_h  = self.data['item_h'].values.astype(np.float32)
        weight  = self.data['weight'].values.astype(np.float32)
        wh_l    = self.data['wh_l'].values.astype(np.float32)
        wh_w    = self.data['wh_w'].values.astype(np.float32)
        wh_h    = self.data['wh_h'].values.astype(np.float32)

        # New columns — fall back gracefully for old CSV files
        af      = self.data['access_freq'].values.astype(np.float32) \
                    if 'access_freq' in self.data.columns else np.ones(n, dtype=np.float32)
        prio    = self.data['priority'].values.astype(np.float32) \
                    if 'priority'    in self.data.columns else np.ones(n, dtype=np.float32)
        door_x  = self.data['door_x'].values.astype(np.float32) \
                    if 'door_x'      in self.data.columns else np.zeros(n, dtype=np.float32)
        door_y  = self.data['door_y'].values.astype(np.float32) \
                    if 'door_y'      in self.data.columns else np.zeros(n, dtype=np.float32)
        # scenario_id used for collision-penalty grouping (items sharing same warehouse)
        self.scenario_id = self.data['scenario_id'].values.astype(np.int64) \
                    if 'scenario_id' in self.data.columns else np.arange(n, dtype=np.int64)

        item_l_max  = max(float(item_l.max()), 1.0)
        item_w_max  = max(float(item_w.max()), 1.0)
        item_h_max  = max(float(item_h.max()), 1.0)
        weight_max  = max(float(weight.max()),  1.0)
        af_max      = max(float(af.max()),       1.0)
        prio_max    = max(float(prio.max()),     1.0)

        item_vol = item_l * item_w * item_h
        wh_vol   = wh_l  * wh_w  * wh_h
        item_area = item_l * item_w
        wh_area   = wh_l  * wh_w

        # Composite ranking key (af × priority) — mirrors optimizer sort key
        af_prio = self.data['af_prio'].values.astype(np.float32) \
                    if 'af_prio' in self.data.columns else (af * prio)
        af_prio_max = max(float(af_prio.max()), 1.0)

        self.x = np.zeros((n, 20), dtype=np.float32)

        # ── Feature layout (must match ml_utils.py exactly) ─────────────────
        self.x[:, 0]  = item_l / item_l_max
        self.x[:, 1]  = item_w / item_w_max
        self.x[:, 2]  = item_h / item_h_max
        self.x[:, 3]  = weight / weight_max
        self.x[:, 4]  = self.data['fragile'].values.astype(np.float32)
        self.x[:, 5]  = self.data['stackable'].values.astype(np.float32)
        self.x[:, 6]  = self.data['can_rotate'].values.astype(np.float32)
        # 7: normalised access frequency — high-AF items should be near door
        self.x[:, 7]  = af / af_max
        # 8: normalised priority — high-prio items should hug zone walls
        self.x[:, 8]  = prio / prio_max
        # 9: door X position normalised to warehouse length
        self.x[:, 9]  = door_x / (wh_l + 1e-6)
        self.x[:, 10] = item_vol / max(item_l_max * item_w_max * item_h_max, 1e-6)
        # 11: door Y position normalised to warehouse width
        self.x[:, 11] = door_y / (wh_w + 1e-6)
        self.x[:, 12] = item_vol / (wh_vol + 1e-6)
        self.x[:, 13] = item_area / max(item_l_max * item_w_max, 1e-6)
        # 14: is non-fragile (1=robust, 0=fragile)
        self.x[:, 14] = (self.data['fragile'].values == 0).astype(np.float32)
        self.x[:, 15] = item_area / (wh_area + 1e-6)
        self.x[:, 16] = item_l / (wh_l + 1e-6)
        self.x[:, 17] = item_w / (wh_w + 1e-6)
        # 18: sequence progress within scenario — use i/N for consistent scaling
        # Group by scenario_id and compute within-scenario fill fraction
        _unique_sids = np.unique(self.scenario_id)
        for _sid in _unique_sids:
            _mask = self.scenario_id == _sid
            _count = int(np.sum(_mask))
            self.x[_mask, 18] = np.arange(_count, dtype=np.float32) / float(max(_count, 1))
        # 19: normalised af×priority composite — directly encodes optimizer sort key
        self.x[:, 19] = af_prio / af_prio_max

        self.y = self.data[['target_x', 'target_y', 'target_z', 'target_rot']].values.astype(np.float32)
        self.y[:, 0] = self.y[:, 0] / (wh_l + 1e-5)
        self.y[:, 1] = self.y[:, 1] / (wh_w + 1e-5)
        self.y[:, 2] = self.y[:, 2] / (wh_h + 1e-5)
        self.y[:, 3] = self.y[:, 3] / 6.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.x[idx]),
                torch.tensor(self.y[idx]),
                torch.tensor(self.scenario_id[idx]))


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

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS,
        persistent_workers=False,
    )

    # Use the module-level device (CUDA if available, set at startup)
    model = PackingModel()
    model.to(device)
    if torch.cuda.is_available():
        print(f"  Device : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB total")
    
    # Weighted Loss: z has higher weight; rot is less critical
    # [x, y, z, rot] weights
    weight_v = torch.tensor([1.0, 1.0, 2.5, 0.5]).to(device)
    def weighted_mse_loss(input, target):
        return (weight_v * (input - target) ** 2).mean()

    def door_proximity_loss(pred, batch_x):
        """
        High-AF items (feature 7 > 0.6, i.e. AF>=6/10) should be placed near the
        door.  Door position is in features 9 (door_x/wh_l) and 11 (door_y/wh_w).
        Penalise the squared distance between prediction and door in normalised space.
        """
        af_norm   = batch_x[:, 7]                          # normalised AF
        door_xn   = batch_x[:, 9]                          # door_x / wh_l
        door_yn   = batch_x[:, 11]                         # door_y / wh_w
        high_af   = (af_norm > 0.6).float()                # items with AF >= 6
        dist_sq   = (pred[:, 0] - door_xn) ** 2 + (pred[:, 1] - door_yn) ** 2
        return (high_af * dist_sq).mean()

    def fragile_z_loss(pred, batch_x):
        """
        Fragile items belong in the upper zone (z > 0.4 normalised).
        Penalise fragile items predicted below that threshold.
        Non-fragile items are penalised if predicted above 0.55 (they should stay
        in the lower zone so fragile items can stack above them).
        """
        fragile  = batch_x[:, 4]                           # 1 = fragile
        nonfrag  = batch_x[:, 14]                          # 1 = non-fragile
        pred_z   = pred[:, 2]

        # Fragile below upper zone → push upward
        frag_penalty   = fragile  * torch.clamp(0.40 - pred_z, min=0.0) ** 2
        # Non-fragile above mid → push downward (keep floor clear for fragile stacking)
        nonfrag_penalty = nonfrag * torch.clamp(pred_z - 0.55, min=0.0) ** 2

        return frag_penalty.mean() + nonfrag_penalty.mean()

    def wall_proximity_loss(pred, batch_x):
        """
        High-priority items (feature 8 > 0.5, i.e. prio >= 2/3) should be placed
        near the nearest wall of the warehouse. Penalise the distance to the
        nearest wall boundary in normalised [0,1] space.
        """
        prio_norm = batch_x[:, 8]                          # normalised priority
        high_prio = (prio_norm > 0.5).float()              # items with prio >= 2
        # Distance to nearest wall in normalised space (pred in [0,1])
        d_left  = pred[:, 0]
        d_right = 1.0 - pred[:, 0]
        d_front = pred[:, 1]
        d_back  = 1.0 - pred[:, 1]
        d_wall  = torch.min(torch.min(d_left, d_right), torch.min(d_front, d_back))
        return (high_prio * d_wall ** 2).mean()

    def calculate_collision_penalty(pred, batch_x, batch_sid):
        """
        Penalizes overlapping bounding boxes within items of the same scenario.
        Optimized: groups by scenario_id and only computes within-group collisions
        to avoid wasting compute on cross-scenario pairs.
        """
        l_prime = batch_x[:, 16].clamp(min=1e-4)
        w_prime = batch_x[:, 17].clamp(min=1e-4)
        h_prime = batch_x[:, 2].clamp(min=1e-4)

        x1, y1, z1 = pred[:, 0], pred[:, 1], pred[:, 2]
        x2 = x1 + l_prime
        y2 = y1 + w_prime
        z2 = z1 + h_prime

        total_overlap = torch.tensor(0.0, device=device)
        total_pairs = 0

        # Group by scenario_id and compute pairwise overlap within each group
        unique_sids = batch_sid.unique()
        for sid in unique_sids:
            mask = batch_sid == sid
            n_g = mask.sum().item()
            if n_g < 2:
                continue
            idx = mask.nonzero(as_tuple=True)[0]

            gx1, gy1, gz1 = x1[idx], y1[idx], z1[idx]
            gx2, gy2, gz2 = x2[idx], y2[idx], z2[idx]

            ix1 = torch.max(gx1.unsqueeze(1), gx1.unsqueeze(0))
            ix2 = torch.min(gx2.unsqueeze(1), gx2.unsqueeze(0))
            iy1 = torch.max(gy1.unsqueeze(1), gy1.unsqueeze(0))
            iy2 = torch.min(gy2.unsqueeze(1), gy2.unsqueeze(0))
            iz1 = torch.max(gz1.unsqueeze(1), gz1.unsqueeze(0))
            iz2 = torch.min(gz2.unsqueeze(1), gz2.unsqueeze(0))

            inter = (torch.clamp(ix2 - ix1, min=0) *
                     torch.clamp(iy2 - iy1, min=0) *
                     torch.clamp(iz2 - iz1, min=0))

            # Exclude self-pairs
            grp_mask = 1.0 - torch.eye(n_g, device=device)
            total_overlap = total_overlap + (inter * grp_mask).sum()
            total_pairs += int(n_g * (n_g - 1))

        return total_overlap / (total_pairs + 1e-6)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_mse": [], "train_coll": [],
               "train_door": [], "train_frag": [], "train_wall": [], "lr": []}

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        # LR warmup: linearly ramp from LR/10 to LR over WARMUP_EPOCHS
        if epoch < WARMUP_EPOCHS:
            warmup_lr = LR * (epoch + 1) / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        total_loss = 0
        total_mse  = 0
        total_coll = 0
        total_door = 0
        total_frag = 0
        total_wall = 0
        n_batches  = 0
        for batch_x, batch_y, batch_sid in train_loader:
            # non_blocking=True overlaps CPU→GPU transfer with GPU compute
            batch_x   = batch_x.to(device, non_blocking=True)
            batch_y   = batch_y.to(device, non_blocking=True)
            batch_sid = batch_sid.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(batch_x)

            mse_loss     = weighted_mse_loss(outputs, batch_y)
            coll_penalty = calculate_collision_penalty(outputs, batch_x, batch_sid)
            door_penalty = door_proximity_loss(outputs, batch_x)
            frag_penalty = fragile_z_loss(outputs, batch_x)
            wall_penalty = wall_proximity_loss(outputs, batch_x)

            # Physics-Informed Loss:
            #   MSE          — primary placement accuracy
            #   collision ×5 — prevent overlapping predictions
            #   door    ×8   — high-AF items must predict near door
            #   fragile ×4   — fragile items must predict upper zone z
            #   wall    ×3   — high-priority items prefer wall adjacency
            loss = (mse_loss
                    + 5.0  * coll_penalty
                    + 8.0  * door_penalty
                    + 4.0  * frag_penalty
                    + 3.0  * wall_penalty)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse  += mse_loss.item()
            total_coll += coll_penalty.item()
            total_door += door_penalty.item()
            total_frag += frag_penalty.item()
            total_wall += wall_penalty.item()
            n_batches  += 1

        avg_train = total_loss / max(n_batches, 1)
        avg_mse   = total_mse  / max(n_batches, 1)
        avg_coll  = total_coll / max(n_batches, 1)
        avg_door  = total_door / max(n_batches, 1)
        avg_frag  = total_frag / max(n_batches, 1)
        avg_wall  = total_wall / max(n_batches, 1)

        # Print VRAM usage after first epoch so we can spot OOM early
        if epoch == 0 and torch.cuda.is_available():
            used_mb  = torch.cuda.memory_allocated(0) // 1024**2
            total_mb = torch.cuda.get_device_properties(0).total_memory // 1024**2
            print(f"  VRAM used after epoch 1: {used_mb} / {total_mb} MB")

        # --- Validate ---
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_y, _sid in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                pred = model(batch_x)
                val_loss += weighted_mse_loss(pred, batch_y).item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)

        # Only step cosine scheduler after warmup is done
        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        # Log history
        history["train_loss"].append(avg_train)
        history["train_mse"].append(avg_mse)
        history["train_coll"].append(avg_coll)
        history["train_door"].append(avg_door)
        history["train_frag"].append(avg_frag)
        history["train_wall"].append(avg_wall)
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
            print(f"  Epoch [{epoch+1:3d}/{EPOCHS}] | "
                  f"Total: {avg_train:.5f} | MSE: {avg_mse:.5f} | "
                  f"Coll: {avg_coll:.5f} | Door: {avg_door:.5f} | "
                  f"Frag: {avg_frag:.5f} | Wall: {avg_wall:.5f} | "
                  f"Val: {avg_val:.5f} (Best: {best_val_loss:.5f})")

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

        # Free GPU cache between runs so each model starts with clean VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
