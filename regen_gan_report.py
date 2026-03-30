
import os
import sys

# Ensure the root directory is in the path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from evaluate_metrics import (
    generate_gan_metrics_report,
    METRICS_BASE_DIR,
    VISUALS_DIR
)

print(f"Regenerating GAN Report at {METRICS_BASE_DIR}...")
generate_gan_metrics_report()
print("Done! Check Documents/04_Machine_Learning/Performance_Metrics/model_metrics_gan.md and visuals.")
