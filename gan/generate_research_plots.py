import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuration
LOSS_HISTORY = 'gan/loss_history.json'
REAL_DATA = 'datasets/datasets.csv'
SYNTHETIC_DATA = 'training_data/warehouse_training.csv'
OUTPUT_DIR = 'Documents/04_Machine_Learning/Performance_Metrics/metrics_visuals/'
EQUILIBRIUM = 0.693147

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_loss_plots():
    print("Generating loss plots...")
    with open(LOSS_HISTORY, 'r') as f:
        history = json.load(f)
    
    epochs = range(len(history['d_loss']))
    
    # 1. Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['d_loss'], label='Discriminator Loss', alpha=0.8)
    plt.plot(epochs, history['g_loss'], label='Generator Loss', alpha=0.8)
    plt.axhline(y=EQUILIBRIUM, color='r', linestyle='--', label='Nash Equilibrium (0.693)')
    plt.title('GAN Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'gan_loss_curves.png'))
    plt.close()

    # 2. Parity Curve (|D-G|)
    parity = [abs(d - g) for d, g in zip(history['d_loss'], history['g_loss'])]
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, parity, color='purple', label='|D_loss - G_loss|')
    plt.title('GAN Parity (Model Harmony)')
    plt.xlabel('Epoch')
    plt.ylabel('Absolute Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'gan_parity_curve.png'))
    plt.close()

    # 3. DTE Curve (Distance to Equilibrium)
    dte_d = [abs(d - EQUILIBRIUM) for d in history['d_loss']]
    dte_g = [abs(g - EQUILIBRIUM) for g in history['g_loss']]
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, dte_d, label='DTE (Discriminator)')
    plt.plot(epochs, dte_g, label='DTE (Generator)')
    plt.title('Distance to Equilibrium (DTE)')
    plt.xlabel('Epoch')
    plt.ylabel('Distance to 0.693')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'gan_dte_curve.png'))
    plt.close()

def load_and_clean_data(file_path, is_real=True):
    # For Real data: columns index 2,3,4,5 are length, width, height, weight
    # For Synthetic data: columns index 0,1,2,3 are length, width, height, weight
    df = pd.read_csv(file_path, header=None, low_memory=False)
    
    if is_real:
        # Check if first row is header
        try:
            float(df.iloc[0, 2])
            start_row = 0
        except:
            start_row = 1
        data = df.iloc[start_row:, [2, 3, 4, 5]]
    else:
        # Check if first row is header (e.g. 'item_l')
        try:
            float(df.iloc[0, 0])
            start_row = 0
        except:
            start_row = 1
        data = df.iloc[start_row:, [0, 1, 2, 3]]
    
    # Rename and clean
    data.columns = ['length', 'width', 'height', 'weight']
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    return data.dropna()

def generate_distribution_plots():
    print("Generating distribution plots...")
    df_real = load_and_clean_data(REAL_DATA, is_real=True)
    df_syn = load_and_clean_data(SYNTHETIC_DATA, is_real=False)

    cols = ['length', 'width', 'height', 'weight']

    # 4. KDE Overlays
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.kdeplot(df_real[col], ax=axes[i], label='Real', fill=True, color='blue', alpha=0.4)
        sns.kdeplot(df_syn[col], ax=axes[i], label='Synthetic', fill=True, color='red', alpha=0.4)
        axes[i].set_title(f'{col.capitalize()} Distribution')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gan_kde_overlays.png'))
    plt.close()

    # 5. PCA Projection
    n_samples = min(2000, len(df_real), len(df_syn))
    df_real_sub = df_real.sample(n_samples, random_state=42)
    df_syn_sub = df_syn.sample(n_samples, random_state=42)
    
    scaler = StandardScaler()
    combined = pd.concat([df_real_sub, df_syn_sub])
    scaled = scaler.fit_transform(combined)
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(coords[:n_samples, 0], coords[:n_samples, 1], color='blue', alpha=0.5, label='Real', s=10)
    plt.scatter(coords[n_samples:, 0], coords[n_samples:, 1], color='red', alpha=0.5, label='Synthetic', s=10)
    plt.title('PCA Projection: Real vs Synthetic Warehouse Items')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gan_pca_projection.png'))
    plt.close()

    # 6. Correlation Delta
    # Select only numeric for correlation to avoid TypeError
    corr_real = df_real.select_dtypes(include=[np.number]).corr()
    corr_syn = df_syn.select_dtypes(include=[np.number]).corr()
    delta = corr_real - corr_syn
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(delta, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('Correlation Delta (Real - Synthetic)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'gan_correlation_delta.png'))
    plt.close()

if __name__ == "__main__":
    generate_loss_plots()
    generate_distribution_plots()
    print("All research plots generated successfully.")
