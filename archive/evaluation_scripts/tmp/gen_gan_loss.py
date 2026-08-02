import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
ASSETS_DIR = r"c:\Users\jebzw\OneDrive\Documents\Github\Training-Bin-Packing\Documents\05_Assets\images"
LOSS_HISTORY_PATH = r"c:\Users\jebzw\OneDrive\Documents\Github\Training-Bin-Packing\gan\loss_history.json"
os.makedirs(ASSETS_DIR, exist_ok=True)

def generate_gan_loss():
    print("Generating Real GAN Loss Curves...")
    
    if not os.path.exists(LOSS_HISTORY_PATH):
        print(f"Error: {LOSS_HISTORY_PATH} not found.")
        return

    with open(LOSS_HISTORY_PATH, 'r') as f:
        history = json.load(f)
    
    d_loss = history["d_loss"]
    g_loss = history["g_loss"]
    epochs = np.arange(1, len(d_loss) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, d_loss, label='Discriminator Loss', color='#e74c3c', linewidth=1.5, marker='o', markersize=4)
    plt.plot(epochs, g_loss, label='Generator Loss', color='#3498db', linewidth=1.5, marker='s', markersize=4)
    
    plt.title('GAN Training Loss Curves (Real Data from fit_ga)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Adversarial Loss')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Shade the "equilibrium" zone (approx 0.5-1.0)
    plt.axhspan(0.5, 1.0, color='gray', alpha=0.1, label='Equilibrium Zone')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "gan_loss_curves.png"), dpi=150)
    plt.close()
    print("Saved gan_loss_curves.png with real data.")

if __name__ == "__main__":
    generate_gan_loss()
