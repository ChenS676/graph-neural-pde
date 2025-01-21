import pandas as pd
import importlib.util

# Load the dictionary from a Python file
def load_dict_from_file(file_path, var_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, var_name)

# Define file paths
input_file = '/hkfs/work/workspace/scratch/cc7738-rebuttal/graph-neural-pde/src/best_params.py'  
output_excel = '/hkfs/work/workspace/scratch/cc7738-rebuttal/graph-neural-pde/src/best_params.xlsx'  

# Variable name of the dictionary in the file
dict_var_name = 'best_params_dict'

# Load the dictionary
best_params_dict = load_dict_from_file(input_file, dict_var_name)

# Convert to DataFrame
df = pd.DataFrame(best_params_dict).T

# Save as Excel
df.to_csv(output_excel, index=True)

print(f"Dictionary has been saved as an Excel file at: {output_excel}")
