import json
import re

NOTEBOOK_PATH = "Step_by_Step_Training.ipynb"

def surgically_update_parameters():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "algorithms = [" in source and "fit_eo_ga" in source:
                
                # First, extract the lines
                lines = source.split('\n')
                for i, line in enumerate(lines):
                    if "'fit_eo_ga'" in line:
                        # Make this one very fast (e.g. ga_generations=1, eo_iterations=2)
                        lines[i] = "        ('fit_eo_ga', HybridOptimizer(ga_generations=1, eo_iterations=2)),"
                    elif "'fit_ga_eo'" in line:
                        # Restore this one to the original
                        lines[i] = "        ('fit_ga_eo', HybridOptimizer(ga_generations=5, eo_iterations=10))"
                
                # Join back with newlines
                new_source = '\n'.join(lines)
                
                # Re-apply to cell source using correct Jupyter format
                cell['source'] = [line + '\n' if j < len(new_source.split('\n')) - 1 else line for j, line in enumerate(new_source.split('\n'))]

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
        print("Parameters updated surgically!")

if __name__ == "__main__":
    surgically_update_parameters()
