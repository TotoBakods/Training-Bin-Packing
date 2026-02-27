import json

NOTEBOOK_PATH = "Step_by_Step_Training.ipynb"

def update_parameters():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "algorithms = [" in source and "fit_eo_ga" in source:
                # Replace the HybridOptimizer parameters
                source = source.replace("HybridOptimizer(ga_generations=5, eo_iterations=10)", 
                                        "HybridOptimizer(ga_generations=2, eo_iterations=5)")
                
                # Re-apply to cell source
                cell['source'] = [line + '\n' if i < len(source.split('\n')) - 1 else line for i, line in enumerate(source.split('\n'))]

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
        print("Parameters updated successfully.")

if __name__ == "__main__":
    update_parameters()
