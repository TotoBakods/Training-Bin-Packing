import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def generate_pareto_frontier():
    """Generates the Speed (Latency) vs Quality (Fitness) Pareto Frontier plot."""
    # Data from latest evaluation results
    results = [
        {"Model": "EO", "Latency": 2555.3, "Fitness": 25.1},
        {"Model": "EO-GA", "Latency": 2814.1, "Fitness": 64.4},
        {"Model": "GA", "Latency": 2819.5, "Fitness": 57.3},
        {"Model": "GA-EO", "Latency": 2921.3, "Fitness": 56.0}
    ]
    
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x="Latency", y="Fitness", hue="Model", s=250, palette="viridis")
    
    # Annotate points
    for i in range(len(df)):
        plt.text(df.Latency[i]+10, df.Fitness[i], df.Model[i], fontsize=12, fontweight='bold')
    
    plt.title("SOTA Performance Frontier: Speed vs. Quality", fontsize=15, fontweight='bold')
    plt.xlabel("Total Inference + Repair Latency (ms)", fontsize=12)
    plt.ylabel("Volumetric Packing Fitness (%)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path = "Documents/04_Machine_Learning/Performance_Metrics/metrics_visuals/pareto_frontier.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def generate_correlation_delta():
    """Generates a Delta Heatmap (Real - Synthetic) for feature correlations."""
    try:
        real_df = pd.read_csv("datasets/datasets.csv")
        synth_df = pd.read_csv("gan/generated_items.csv")
        
        cols = ['length', 'width', 'height', 'weight']
        real_corr = real_df[cols].corr()
        synth_corr = synth_df[cols].corr()
        
        delta_corr = real_corr - synth_corr
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(delta_corr, annot=True, cmap="coolwarm", center=0, fmt=".4f")
        plt.title("Correlation Delta: Real - Synthetic\n(SOTA Fidelity Audit)", fontsize=13, fontweight='bold')
        
        output_path = "Documents/04_Machine_Learning/Performance_Metrics/metrics_visuals/gan_correlation_delta.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Skipping Correlation Delta: {e}")

def generate_pca_projection():
    """Generates a PCA projection of Real vs Synthetic samples."""
    try:
        real_df = pd.read_csv("datasets/datasets.csv").head(1000)
        synth_df = pd.read_csv("gan/generated_items.csv").head(1000)
        
        cols = ['length', 'width', 'height', 'weight']
        real_data = real_df[cols].values
        synth_data = synth_df[cols].values
        
        combined = np.vstack([real_data, synth_data])
        pca = PCA(n_components=2)
        proj = pca.fit_transform(combined)
        
        plt.figure(figsize=(10, 7))
        plt.scatter(proj[:1000, 0], proj[:1000, 1], alpha=0.4, label="Real Distribution", s=30, color='blue')
        plt.scatter(proj[1000:, 0], proj[1000:, 1], alpha=0.3, label="GAN Distribution", s=30, color='red', marker='x')
        
        plt.title("SOTA PCA Projection: Distribution Overlap", fontsize=15, fontweight='bold')
        plt.legend()
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        
        output_path = "Documents/04_Machine_Learning/Performance_Metrics/metrics_visuals/gan_pca_projection.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Skipping PCA Projection: {e}")

if __name__ == "__main__":
    generate_pareto_frontier()
    generate_correlation_delta()
    generate_pca_projection()
