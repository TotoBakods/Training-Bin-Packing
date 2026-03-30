import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

import evaluate_metrics

if __name__ == "__main__":
    print("Starting enhanced reporting only run...")
    # This will generate the enhanced SKU diversity plot and update the GAN metrics report with samples
    evaluate_metrics.generate_gan_metrics_report()
    print("Done!")
