# ML Packing Model

The ML Packing Model is a neural network-based optimization engine that predicts the optimal position (X, Y, Z, Rotation) of an item based on its dimensions, weight, and properties.

## Model Architecture
The model is a deep multi-layered perceptron with batch normalization, leaky ReLU activation, and dropout for regularization. It takes 18 features as input and outputs 4 values representing the predicted coordinates and rotation.

## Inputs (18 Features)
- **Basic Functions**: Length, Width, Height, Weight, Fragility, Stackable...
- **Advanced Features**: Item Volume, Warehouse Volume, Item Area, Warehouse Area...

## Code Snippet (PyTorch Implementation)

```python
import torch
import torch.nn as nn

class PackingModel(nn.Module):
    def __init__(self, input_dim=18, output_dim=4):
        super(PackingModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),

            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)

# Example training loss:
def weighted_mse_loss(input, target):
    """Wait and bias X/Y coordinate predictions to reduce displacement."""
    weight_v = torch.tensor([2.0, 2.0, 1.0, 1.0]) # Weights: X=2.0, Y=2.0, Z=1.0, R=1.0
    return (weight_v * (input - target) ** 2).mean()

---

For more details on the input preprocessing and output scaling, see the [Feature Engineering & Normalization Pipeline](feature_engineering_pipeline.md).
```
