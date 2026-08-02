import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
import sys

# Add current dir to sys path to import PackingModel
sys.path.append(os.getcwd())
from ml_utils import PackingModel

# Paths
ASSETS_DIR = r"c:\Users\jebzw\OneDrive\Documents\Github\Training-Bin-Packing\Documents\05_Assets\images"
MODEL_PATH = r"c:\Users\jebzw\OneDrive\Documents\Github\Training-Bin-Packing\models\model_fit_ga.pth"
DATA_PATH = r"c:\Users\jebzw\OneDrive\Documents\Github\Training-Bin-Packing\training_data\fit_ga.csv"

def generate_pybullet_figures():
    print("Generating PyBullet figures...")
    device = torch.device('cpu')
    model = PackingModel()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    df = pd.read_csv(DATA_PATH, nrows=100)
    sample = df.iloc[5]
    
    l, w, h = sample['item_l'], sample['item_w'], sample['item_h']
    wh_l, wh_w, wh_h = sample['wh_l'], sample['wh_w'], sample['wh_h']
    
    iv, ia = l*w*h, l*w
    wv, wa = wh_l*wh_w*wh_h, wh_l*wh_w
    features = [
        l/10.0, w/10.0, h/10.0,
        sample['weight']/100.0, float(sample['fragile']), float(sample['stackable']), float(sample['can_rotate']),
        wh_l/100.0, wh_w/100.0, wh_h/100.0,
        iv/10.0, wv/1000.0, iv/(wv+1e-6),
        ia/10.0, wa/100.0, ia/(wa+1e-6),
        l/(wh_l+1e-6), w/(wh_w+1e-6)
    ]
    
    with torch.no_grad():
        inputs = torch.tensor([features], dtype=torch.float32)
        outputs = model(inputs).numpy()[0]
        
    pred_x, pred_y, pred_z = outputs[0]*wh_l, outputs[1]*wh_w, outputs[2]*wh_h
    target_x, target_y, target_z = sample['target_x'], sample['target_y'], sample['target_z']

    # Start PyBullet
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    # --- Figure 13 ---
    t_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[l/2, w/2, h/2], rgbaColor=[0, 1, 0, 0.4])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=t_v, basePosition=[target_x, target_y, target_z + h/2])
    
    p_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[l/2, w/2, h/2], rgbaColor=[1, 0, 0, 0.7])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=p_v, basePosition=[pred_x, pred_y, pred_z + h/2])

    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[target_x, target_y, target_z], distance=3.0, yaw=45, pitch=-35, roll=0, upAxisIndex=2)
    proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0)
    
    w_img, h_img, rgb, depth, mask = p.getCameraImage(512, 512, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER)
    # Convert to uint8 for saving
    rgb_array = np.reshape(rgb, (h_img, w_img, 4)).astype(np.uint8)
    plt.imsave(os.path.join(ASSETS_DIR, "regression_accuracy.png"), rgb_array)
    print("Saved regression_accuracy.png")

    # --- Figure 14 ---
    p.resetSimulation()
    p.loadURDF("plane.urdf")
    
    for i in range(min(len(df), 20)):
        r = df.iloc[i]
        rl, rw, rh = r['item_l'], r['item_w'], r['item_h']
        rx, ry, rz = r['target_x'], r['target_y'], r['target_z']
        v = p.createVisualShape(p.GEOM_BOX, halfExtents=[rl/2, rw/2, rh/2], rgbaColor=[np.random.rand(), np.random.rand(), np.random.rand(), 1])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=v, basePosition=[rx, ry, rz + rh/2])

    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[wh_l/2, wh_w/2, 1], distance=8, yaw=30, pitch=-40, roll=0, upAxisIndex=2)
    w_img, h_img, rgb, depth, mask = p.getCameraImage(640, 480, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER)
    rgb_array = np.reshape(rgb, (h_img, w_img, 4)).astype(np.uint8)
    plt.imsave(os.path.join(ASSETS_DIR, "pybullet_validation.png"), rgb_array)
    print("Saved pybullet_validation.png")
    
    p.disconnect()

if __name__ == "__main__":
    generate_pybullet_figures()
