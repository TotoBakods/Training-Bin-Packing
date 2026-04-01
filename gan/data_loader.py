import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class WarehouseItemDataset(Dataset):
    def __init__(self, csv_file):
        # Load data using Pandas
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: File {csv_file} not found.")
            df = pd.DataFrame(columns=['length', 'width', 'height', 'weight'])

        # Features: length, width, height, weight
        # Ensure extraction deals with potential missing values or bad types
        features = df[['length', 'width', 'height', 'weight']].apply(pd.to_numeric, errors='coerce').dropna()
        
        # Filter positive values
        features = features[
            (features['length'] > 0) & 
            (features['width'] > 0) & 
            (features['height'] > 0) & 
            (features['weight'] > 0)
        ]

        if features.empty:
            print("Warning: No valid items found in dataset!")
            # Dummy data to prevent crash
            self.data = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        else:
            self.data = features.values.astype(np.float32)

        # Normalize data to [0, 1] range
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])
    
    def get_scaler(self):
        return self.scaler

def get_dataloaders(csv_file, batch_size=32, val_split=0.2):
    dataset = WarehouseItemDataset(csv_file)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    return train_loader, val_loader, dataset.get_scaler()

def get_gpu_dataset(csv_file, device):
    """Loads the entire dataset directly onto the GPU for maximum training speed."""
    dataset = WarehouseItemDataset(csv_file)
    data_tensor = torch.from_numpy(dataset.data).to(device)
    return data_tensor, dataset.get_scaler()
