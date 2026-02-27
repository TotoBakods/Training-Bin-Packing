import json
import ast

with open("Step_by_Step_Training.ipynb", 'r', encoding='utf-8') as f:
    nb = json.load(f)
    
code_text = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # strip magics
        source = "\n".join([line for line in source.split('\n') if not line.startswith('!')])
        code_text += source + "\n\n"

try:
    ast.parse(code_text)
    print("Syntax verification passed!")
except SyntaxError as e:
    print(f"Syntax error at line {e.lineno}: {e.text}")
