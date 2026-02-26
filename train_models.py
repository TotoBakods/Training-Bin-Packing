import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import glob
import pickle
from ml_utils import PackingModel

# Configuration
DATA_DIR = "training_data"
MODELS_DIR = "models"
EPOCHS = 50 
BATCH_SIZE = 64
LR = 0.001

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

class WarehouseDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
        # Features: item dims (3), weight (1), flags (3), wh dims (3)
        # We normalize inputs roughly
        self.x = self.data[['item_l', 'item_w', 'item_h', 'weight', 'fragile', 'stackable', 'can_rotate', 'wh_l', 'wh_w', 'wh_h']].values.astype(np.float32)
        
        # Normalize dimensions relative to warehouse? 
        # For now simple scaling. max dim mostly < 100.
        self.x[:, 0:3] /= 10.0 # items usually small
        self.x[:, 7:10] /= 100.0 # warehouse dims
        
        # Targets: x, y, z, rot
        # Normalize x,y,z by warehouse dims
        self.y = self.data[['target_x', 'target_y', 'target_z', 'target_rot']].values.astype(np.float32)
        
        # Avoid division by zero
        wh_l = self.data['wh_l'].values.astype(np.float32) + 1e-5
        wh_w = self.data['wh_w'].values.astype(np.float32) + 1e-5
        wh_h = self.data['wh_h'].values.astype(np.float32) + 1e-5
        
        self.y[:, 0] = self.y[:, 0] / wh_l
        self.y[:, 1] = self.y[:, 1] / wh_w
        self.y[:, 2] = self.y[:, 2] / wh_h
        # Rotation: 0, 90, 180, 270 (approx codes 0-3 for flat). optimize.py uses 0-5.
        # Normalize rotation to 0-1 range (div by 6)
        self.y[:, 3] = self.y[:, 3] / 6.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

# PackingModel imported from ml_utils


def train_model(csv_path, model_name):
    print(f"Training model for {csv_path}...")
    dataset = WarehouseDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = PackingModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.6f}")

    # Save
    save_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

def run_training():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("No CSV files found in training_data/")
        return

    for csv_file in csv_files:
        # data_fit_eo.csv -> model_fit_eo
        basename = os.path.basename(csv_file) # fit_eo.csv
        name = os.path.splitext(basename)[0] # fit_eo
        train_model(csv_file, f"model_{name}")

if __name__ == "__main__":
    run_training()
