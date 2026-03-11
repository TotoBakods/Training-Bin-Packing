import os
from train_models import train_model

DATA_DIR = "training_data"
csv_path = os.path.join(DATA_DIR, "fit_eo_ga.csv")

# We want faster but good quality
# the train_models.py defines EPOCHS=50, BATCH_SIZE=64 globally. We will patch them dynamically.
import train_models
train_models.EPOCHS = 20
train_models.BATCH_SIZE = 128
train_models.LR = 0.002

if os.path.exists(csv_path):
    print("Training fit_eo_ga...")
    train_models.train_model(csv_path, "model_fit_eo_ga")
    print("Done.")
else:
    print(f"Data not found: {csv_path}")
