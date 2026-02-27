import json

NOTEBOOK_PATH = "Step_by_Step_Training.ipynb"

def fix_syntax():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            for i, line in enumerate(source):
                if line == "        ('fit_eo_ga', HybridOptimizer(ga_generations=1, eo_iterations=2)),\n" and "try:\n" in source[i-1]:
                    # This is the erroneously replaced line
                    source[i] = "                    if 'Hybrid' in str(type(optimizer)) and algo_name == 'fit_eo_ga':\n"

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
        print("Syntax fixed!")

if __name__ == "__main__":
    fix_syntax()
