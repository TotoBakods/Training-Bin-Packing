import json
import os

NOTEBOOK_PATH = "Step_by_Step_Training.ipynb"

def patch_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb['cells']
    
    # 1. Update pip install cell
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "!pip install torch" in source:
                if "scikit-learn" not in source:
                    source = source.replace("flask-cors", "flask-cors scikit-learn")
                    cell['source'] = [line + '\n' if i < len(source.split('\n')) - 1 else line for i, line in enumerate(source.split('\n'))]
                break

    # 2. Insert Cells after import torch cell
    insert_idx = -1
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "import torch" in source and "os.makedirs" in source:
                insert_idx = i + 1
                break
                
    if insert_idx != -1:
        new_md_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1.5 Prepare Dataset & Train GAN\n",
                "\n",
                "First, we convert the raw `bed-bpp` JSON dataset into a CSV format suitable for training the Generative Adversarial Network (GAN). Then, we train the GAN to learn the distributions of real-world items."
            ]
        }
        
        new_code_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python tools/convert_dataset.py\n",
                "!python gan/train.py"
            ]
        }
        
        # Avoid double-insertion if ran multiple times
        if "Prepare Dataset & Train GAN" not in "".join(cells[insert_idx]['source']):
            cells.insert(insert_idx, new_md_cell)
            cells.insert(insert_idx + 1, new_code_cell)

    # 3. Modify Data Generation Cell
    gan_gen_code = """
import sys
if './gan' not in sys.path:
    sys.path.append('./gan')
import pickle
import random
try:
    from model import Generator
except ImportError:
    from gan.model import Generator

def generate_gan_items(count, warehouse_dims):
    l_wh, w_wh, h_wh = warehouse_dims
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    scaler_path = 'gan/scaler.pkl'
    ckpt_path = 'gan/checkpoints/generator.pth'
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    model = Generator(100, 4).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    z = torch.randn(count, 100).to(device)
    with torch.no_grad():
        gen_data = model(z).cpu().numpy()
        
    original_data = scaler.inverse_transform(gen_data)
    
    items = []
    categories = ['General', 'Electronics', 'Clothing']
    fragile_categories = {'Electronics'}
    
    for i in range(count):
        l, w, h, weight = original_data[i]
        l, w, h, weight = abs(l)*2.0, abs(w)*2.0, abs(h)*2.0, abs(weight)*2.0
        
        cat = random.choice(categories)
        is_fragile = 1 if cat in fragile_categories else 0
        is_stackable = 0 if is_fragile else (1 if random.random() > 0.1 else 0)
        can_rotate = 0 if h > 2 * min(l, w) else 1
        
        items.append({
            'id': str(uuid.uuid4()),
            'length': round(float(l), 2), 'width': round(float(w), 2), 'height': round(float(h), 2),
            'weight': round(float(weight), 2), 'category': cat,
            'can_rotate': can_rotate, 'stackable': is_stackable, 'fragility': is_fragile, 'access_freq': random.randint(1,10),
            'x': 0, 'y': 0, 'z': 0, 'rotation': 0
        })
    return items
"""

    for cell in cells:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "def generate_random_items(" in source:
                # Replace definition
                import re
                source = re.sub(
                    r"def generate_random_items\(count, warehouse_dims\):.*?(?=def run_data_generation)",
                    gan_gen_code + "\n",
                    source,
                    flags=re.DOTALL
                )
                
                # Replace call
                source = source.replace(
                    "items = generate_random_items(ITEMS_PER_SAMPLE, (wh_l, wh_w, wh_h))",
                    "items = generate_gan_items(ITEMS_PER_SAMPLE, (wh_l, wh_w, wh_h))"
                )
                
                cell['source'] = [line + '\n' if i < len(source.split('\n')) - 1 else line for i, line in enumerate(source.split('\n'))]

    # 4. Modify Evaluation Cell
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "def evaluate_model(" in source:
                source = source.replace(
                    "items = generate_random_items(20, (wh_l, wh_w, wh_h))",
                    "items = generate_gan_items(20, (wh_l, wh_w, wh_h))"
                )
                cell['source'] = [line + '\n' if i < len(source.split('\n')) - 1 else line for i, line in enumerate(source.split('\n'))]

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
        print("Notebook patched successfully.")

if __name__ == "__main__":
    patch_notebook()
