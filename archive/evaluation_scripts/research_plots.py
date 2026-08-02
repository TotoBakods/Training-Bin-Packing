import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def generate_comparison_plots():
    """Generates comparative bar charts for PSR, SSR, and VU across variants."""
    try:
        df = pd.read_csv("gan/research_comparison_metrics.csv")
        
        # 1. PSR Comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Scale", y="PSR (%)", hue="Variant", palette="mako")
        plt.title("Placement Success Rate (PSR) Comparison", fontsize=14, fontweight='bold')
        plt.ylim(0, 110)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("Documents/04_Machine_Learning/Performance_Metrics/metrics_visuals/psr_comparison.png", dpi=150)
        plt.close()

        # 2. SSR Comparison (Stability)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Scale", y="SSR (%)", hue="Variant", palette="rocket")
        plt.title("Support Surface Ratio (SSR) - Physical Stability", fontsize=14, fontweight='bold')
        plt.ylim(0, 110)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("Documents/04_Machine_Learning/Performance_Metrics/metrics_visuals/ssr_comparison.png", dpi=150)
        plt.close()

        # 3. VU Comparison (Utilization)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Scale", y="VU (%)", hue="Variant", palette="viridis")
        plt.title("Volumetric Utilization (VU) Performance", fontsize=14, fontweight='bold')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("Documents/04_Machine_Learning/Performance_Metrics/metrics_visuals/vu_comparison.png", dpi=150)
        plt.close()
        
        print("Successfully generated comparison plots: psr, ssr, vu.")
    except Exception as e:
        print(f"Error generating comparison plots: {e}")

def generate_pareto_frontier_dynamic():
    """Generates the Speed vs Quality Pareto Frontier using real data."""
    try:
        df = pd.read_csv("gan/research_comparison_metrics.csv")
        # Filter for 600 items (most complex)
        df_600 = df[df['Scale'] == 600]
        
        plt.figure(figsize=(10, 7))
        # We use VU as quality and Latency as speed
        sns.scatterplot(data=df_600, x="Latency (ms)", y="VU (%)", hue="Variant", s=300, palette="deep")
        
        for i, row in df_600.iterrows():
            plt.text(row['Latency (ms)']+50, row['VU (%)'], row['Variant'], fontsize=10, fontweight='bold')
            
        plt.title("Performance Frontier (600 Items): Speed vs. Quality", fontsize=15, fontweight='bold')
        plt.xlabel("Total Inference + Repair Latency (ms)", fontsize=12)
        plt.ylabel("Volumetric Packing Efficiency (%)", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        plt.savefig("Documents/04_Machine_Learning/Performance_Metrics/metrics_visuals/pareto_frontier.png", dpi=150)
        plt.close()
        print("Generated: pareto_frontier.png")
    except Exception as e:
        print(f"Error generating pareto frontier: {e}")

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs("Documents/04_Machine_Learning/Performance_Metrics/metrics_visuals", exist_ok=True)
    
    generate_comparison_plots()
    generate_pareto_frontier_dynamic()
    generate_correlation_delta()
    generate_pca_projection()
