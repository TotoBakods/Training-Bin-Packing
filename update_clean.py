import json
import re

NOTEBOOK_PATH = "Step_by_Step_Training.ipynb"

def update_algorithms():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "algorithms = [" in source:
                
                # Replace the exact block
                import re
                new_source = re.sub(
                    r"algorithms = \[.*?\]",
                    "algorithms = [\n"
                    "        ('fit_eo', ExtremalOptimization(iterations=20)),\n"
                    "        ('fit_ga', GeneticAlgorithm(population_size=20, generations=10)),\n"
                    "        ('fit_eo_ga', HybridOptimizer(ga_generations=1, eo_iterations=2)),\n"
                    "        ('fit_ga_eo', HybridOptimizer(ga_generations=5, eo_iterations=10))\n"
                    "    ]",
                    source,
                    flags=re.DOTALL
                )
                
                cell['source'] = [line + '\n' if j < len(new_source.split('\n')) - 1 else line for j, line in enumerate(new_source.split('\n'))]

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
        print("Algorithms block replaced cleanly.")

if __name__ == "__main__":
    update_algorithms()
