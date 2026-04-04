import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Paths
HISTORY_PATH = r"c:\Users\jebzw\OneDrive\Documents\Github\Training-Bin-Packing\Documents\04_Machine_Learning\Performance_Metrics\ml_training_history.json"
METRICS_PATH = r"c:\Users\jebzw\OneDrive\Documents\Github\Training-Bin-Packing\Documents\04_Machine_Learning\Performance_Metrics\full_run_metrics.json"
OUTPUT_DIR = r"c:\Users\jebzw\OneDrive\Documents\Github\Training-Bin-Packing\Documents\04_Machine_Learning\Performance_Metrics\metrics_visuals"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_convergence_plots():
    if not os.path.exists(HISTORY_PATH):
        print("History file not found.")
        return

    with open(HISTORY_PATH, "r") as f:
        data = json.load(f)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    for model_name, results in data.items():
        history = results.get("history", {})
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        
        if train_loss:
            plt.plot(train_loss, label=f"{model_name} (Train)", linestyle="--")
        if val_loss:
            plt.plot(val_loss, label=f"{model_name} (Val)", linewidth=2)

    plt.title("Model Convergence: Training vs Validation Loss", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss_curves.png"), dpi=300)
    plt.close()
    print("Generated training_loss_curves.png")

def generate_comparison_plots():
    if not os.path.exists(METRICS_PATH):
        print("Metrics file not found.")
        return

    with open(METRICS_PATH, "r") as f:
        data = json.load(f)

    # Algorithm Comparison (Head-to-Head)
    # Extract metrics for Algorithm performance
    # This is based on Section 4.5 logic in evaluate_metrics.py
    
    algorithms = ["EO", "EO_GA", "GA", "GA_EO"]
    # Mock data if not found, but we'll try to extract real ones
    # For now, let's look at the structure of full_run_metrics.json
    
    # Let's extract PSR and VU for 200, 400, 600 items
    scenarios = ["200_items.csv", "400_items.csv", "600_items.csv"]
    inference_results = data.get("inference_results", {})
    
    plot_data = []
    for scenario in scenarios:
        results = inference_results.get(scenario, {})
        for model_name, metrics in results.items():
            # cleaner model name
            m_label = model_name.replace("model_fit_", "").upper()
            plot_data.append({
                "Scenario": scenario.replace(".csv", ""),
                "Model": m_label,
                "PSR (%)": metrics.get("psr_pct", metrics.get("in_bounds", 0) * 100),
                "VU (Volume %)": metrics.get("su_pct", 0)
            })
    
    df = pd.DataFrame(plot_data)
    if df.empty:
        print("No comparison data found.")
        return

    # PSR Comparison Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Scenario", y="PSR (%)", hue="Model", palette="viridis")
    plt.title("Placement Success Rate (PSR) Across Scales", fontsize=14, fontweight='bold')
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "psr_comparison.png"), dpi=300)
    plt.close()
    
    # VU Comparison Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Scenario", y="VU (Volume %)", hue="Model", palette="magma")
    plt.title("Volumetric Utility (VU) Benchmarks", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "vu_benchmarks.png"), dpi=300)
    plt.close()
    print("Generated comparison plots")

if __name__ == "__main__":
    generate_convergence_plots()
    generate_comparison_plots()
