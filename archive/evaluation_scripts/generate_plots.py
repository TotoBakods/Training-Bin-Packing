import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Paths
GAN_HISTORY_PATH = os.path.join("gan", "loss_history.json")
ML_HISTORY_PATH = os.path.join("Documents", "04_Machine_Learning", "Performance_Metrics", "ml_training_history.json")
OUTPUT_DIR = os.path.join("Documents", "04_Machine_Learning", "Performance_Metrics", "metrics_visuals")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_gan_history():
    if not os.path.exists(GAN_HISTORY_PATH):
        print(f"GAN history not found at {GAN_HISTORY_PATH}")
        return

    with open(GAN_HISTORY_PATH, 'r') as f:
        history = json.load(f)

    epochs = range(len(history["d_loss"]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["d_loss"], label='Discriminator Loss', alpha=0.7)
    plt.plot(epochs, history["g_loss"], label='Generator Loss', alpha=0.7)
    plt.axhline(y=0.693, color='r', linestyle='--', label='Nash Equilibrium (0.693)')
    plt.title('GAN Training Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "gan_loss_curves.png"))
    plt.close()
    
    # Parity Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["parity"], color='purple', label='D/G Parity')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Stability Threshold (0.05)')
    plt.title('GAN Adversarial Parity')
    plt.xlabel('Epoch')
    plt.ylabel('|D_loss - G_loss|')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "gan_parity_curve.png"))
    plt.close()
    
    print("GAN plots generated.")

def plot_ml_history():
    if not os.path.exists(ML_HISTORY_PATH):
        print(f"ML history not found at {ML_HISTORY_PATH}")
        return

    with open(ML_HISTORY_PATH, 'r') as f:
        all_histories = json.load(f)

    for model_name, data in all_histories.items():
        history = data["history"]
        epochs = range(len(history["train_loss"]))
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history["train_loss"], label='Train Loss')
        plt.plot(epochs, history["val_loss"], label='Val Loss')
        plt.title(f'Training Progress: {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Weighted MSE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_learning_curve.png"))
        plt.close()

    print("Packing model plots generated.")

if __name__ == "__main__":
    plot_gan_history()
    plot_ml_history()
