import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import os
import json

real_path = r'c:\Users\jebzw\OneDrive\Documents\Github\Training-Bin-Packing\datasets\datasets.csv'
synth_path = r'c:\Users\jebzw\OneDrive\Documents\Github\Training-Bin-Packing\gan\generated_items.csv'

if not os.path.exists(real_path) or not os.path.exists(synth_path):
    print("Files not found.")
    exit()

df_real = pd.read_csv(real_path)
df_synth = pd.read_csv(synth_path)

features = ['length', 'width', 'height', 'weight']
df_real = df_real[features].dropna()
df_real = df_real[(df_real > 0).all(axis=1)]

# Ensure same columns
df_synth = df_synth[features].apply(pd.to_numeric, errors='coerce').dropna()

stats = {}
for f in features:
    w = wasserstein_distance(df_real[f], df_synth[f])
    stats[f] = {
        'real_mean': float(df_real[f].mean()),
        'real_std': float(df_real[f].std()),
        'synth_mean': float(df_synth[f].mean()),
        'synth_std': float(df_synth[f].std()),
        'wasserstein': float(w)
    }

print(json.dumps(stats, indent=4))
