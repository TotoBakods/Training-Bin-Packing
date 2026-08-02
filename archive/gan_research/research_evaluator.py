import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from gan.model import Generator

# Configuration
LATENT_DIM = 100
OUTPUT_DIM = 4  # Length, Width, Height, Weight
SAMPLES_TO_GENERATE = 5000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
REAL_DATA_PATH = os.path.join('datasets', 'datasets.csv')
GENERATOR_PATH = os.path.join('gan', 'checkpoints', 'generator_best_parity.pth')
SCALER_PATH = os.path.join('gan', 'scaler.pkl')
VISUALS_DIR = os.path.join('Documents', '04_Machine_Learning', 'Performance_Metrics', 'metrics_visuals')
os.makedirs(VISUALS_DIR, exist_ok=True)

def load_real_data():
    df = pd.read_csv(REAL_DATA_PATH)
    features = df[['length', 'width', 'height', 'weight']].apply(pd.to_numeric, errors='coerce').dropna()
    features = features[(features > 0).all(axis=1)]
    return features

def generate_synthetic_data(scaler):
    generator = Generator(LATENT_DIM, OUTPUT_DIM).to(DEVICE)
    generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(SAMPLES_TO_GENERATE, LATENT_DIM).to(DEVICE)
        synth_normalized = generator(z).cpu().numpy()
    
    synth_denormalized = scaler.inverse_transform(synth_normalized)
    return pd.DataFrame(synth_denormalized, columns=['length', 'width', 'height', 'weight'])

def plot_kde_overlays(real_df, synth_df):
    """Generates marginal distribution overlays (KDE) for each feature."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    features = ['length', 'width', 'height', 'weight']
    colors = ['#1f77b4', '#ff7f0e'] # Blue for Real, Orange for Synthetic
    
    for i, feat in enumerate(features):
        sns.kdeplot(real_df[feat], ax=axes[i], fill=True, color=colors[0], label='Real (BED-BPP)', alpha=0.5)
        sns.kdeplot(synth_df[feat], ax=axes[i], fill=True, color=colors[1], label='Synthetic (GAN)', alpha=0.5)
        axes[i].set_title(f"Distribution Consistency: {feat.capitalize()}", fontsize=12, fontweight='bold')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "gan_kde_overlays.png"))
    plt.close()

def plot_correlation_comparison(real_df, synth_df):
    """Generates side-by-side correlation heatmaps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    real_corr = real_df.corr()
    synth_corr = synth_df.corr()
    
    sns.heatmap(real_corr, annot=True, cmap='RdBu_r', center=0, ax=ax1)
    ax1.set_title("Real Relationship Matrix (BED-BPP)", fontsize=14, fontweight='bold')
    
    sns.heatmap(synth_corr, annot=True, cmap='RdBu_r', center=0, ax=ax2)
    ax2.set_title("Synthetic Relationship Matrix (GAN)", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "gan_correlation_fidelity.png"))
    plt.close()

def run_c2st_test(real_df, synth_df):
    """Classifier Two-Sample Test (C2ST) using Random Forest."""
    # Label data
    real_df = real_df.copy()
    synth_df = synth_df.copy()
    real_df['target'] = 1
    synth_df['target'] = 0
    
    combined = pd.concat([real_df, synth_df], axis=0)
    X = combined.drop('target', axis=1)
    y = combined['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Probabilities for positive class (Real)
    probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    return auc

def run_dcr_analysis(real_df, synth_df):
    """Distance to Closest Record (DCR) to check for memorization/overfitting."""
    # Sample a subset for performance if needed, but 1000 is small enough
    # Normalize for fair distance comparison
    real_vals = real_df.values
    synth_vals = synth_df.values
    
    # Normalize both relative to real data max/min for DCR metric
    max_val = real_vals.max(axis=0)
    min_val = real_vals.min(axis=0)
    
    real_norm = (real_vals - min_val) / (max_val - min_val)
    synth_norm = (synth_vals - min_val) / (max_val - min_val)
    
    # For each synth record, find distance to nearest real record
    distances = cdist(synth_norm, real_norm, metric='euclidean')
    min_distances = distances.min(axis=1)
    
    return {
        "mean_dcr": float(min_distances.mean()),
        "min_dcr": float(min_distances.min()),
        "median_dcr": float(np.median(min_distances))
    }

def main():
    print("🚀 Initializing GAN Research Evaluation Engine...")
    
    # 1. Load Data
    real_df = load_real_data()
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    # 2. Generate Synthetic Data
    synth_df = generate_synthetic_data(scaler)
    
    # 3. Visual Evaluations
    print("📈 Generating KDE Marginal Overlays...")
    plot_kde_overlays(real_df, synth_df)
    
    print("📊 Generating Correlation Fidelity Heatmaps...")
    plot_correlation_comparison(real_df, synth_df)
    
    # 4. Statistical Utility & Privacy
    print("🔍 Running C2ST (Classifier Two-Sample Test)...")
    c2st_auc = run_c2st_test(real_df, synth_df)
    
    print("🛡️ Running DCR (Distance to Closest Record) Privacy Check...")
    dcr_stats = run_dcr_analysis(real_df, synth_df)
    
    results = {
        "c2st_auc": c2st_auc,
        "dcr_stats": dcr_stats,
        "sample_size": SAMPLES_TO_GENERATE,
        "status": "V-PASS" if 0.45 <= c2st_auc <= 0.65 else "WARNING"
    }
    
    print(f"\n✅ Evaluation Complete.")
    print(f"C2ST AUC: {c2st_auc:.4f} (Ideal: 0.50)")
    print(f"Mean DCR: {dcr_stats['mean_dcr']:.4f}")
    
    with open(os.path.join('gan', 'research_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to gan/research_metrics.json")

if __name__ == "__main__":
    main()
