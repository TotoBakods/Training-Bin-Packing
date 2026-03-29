# GAN Synthesis Algorithm

The GAN Synthesis algorithm uses a Generative Adversarial Network to create synthetic item data that mirrors real-world distributions. This allows for the generation of large training datasets that are representative of real warehouse inventory.

## Generator Model
The generator takes a noise vector (latent dimension 100) and transforms it into item dimensions (length, width, height) and weight.

## Synthesis Process
1.  **Noise Input**: Random Gaussian noise is fed into the generator.
2.  **Inference**: The generator predicts normalized item properties.
3.  **Inverse Scaling**: Values are transformed back to their original physical units.
4.  **Heuristics**: Post-processing logic assigns category-based properties like fragility and stackability.

## Code Snippet (PyTorch Implementation)

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    """Generator model for synthetic item generation."""
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, output_dim),
        )

    def forward(self, z):
        return self.net(z)

def generate_synthetic_items(n_items=1000):
    """
    Generate synthetic inventory items.
    
    1. Sample latent space Z
    2. G(Z) -> Gated Item Dimensions
    3. Category assignment using weighted distributions
    4. Heuristic-based fragility/stackability logic
    """
    z = torch.randn(n_items, 100)
    with torch.no_grad():
        generated_data = model(z).cpu().numpy()
        
    # Scale back to real-world units
    original_scale_data = scaler.inverse_transform(generated_data)
    
    # post-process
    # ...
    return items
```
