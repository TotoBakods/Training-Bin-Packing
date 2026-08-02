import pandas as pd
import numpy as np
import os

def check_stats():
    real_path = 'datasets/datasets.csv'
    syn_path = 'gan/generated_items.csv'
    
    if not os.path.exists(real_path) or not os.path.exists(syn_path):
        print("Missing data files.")
        return

    df_real = pd.read_csv(real_path)
    df_syn = pd.read_csv(syn_path)
    
    cols = ['length', 'width', 'height', 'weight']
    
    print("| Feature | Real Mean | Synth (Raw) | Synth (Scaled/2) | Fidelity |")
    print("|:---|:---:|:---:|:---:|:---:|")
    
    for col in cols:
        r_m = df_real[col].mean()
        s_m_raw = df_syn[col].mean()
        s_m_adj = s_m_raw / 2.0
        
        fid = max(0, 1 - abs(r_m - s_m_adj)/r_m) * 100
        print(f"| **{col}** | {r_m:.4f} | {s_m_raw:.4f} | {s_m_adj:.4f} | **{fid:.1f}%** |")

if __name__ == "__main__":
    check_stats()
