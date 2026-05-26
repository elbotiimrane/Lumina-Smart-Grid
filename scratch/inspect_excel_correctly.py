import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from prediction.graph_builder import clean_load_excel

data_dir = "data"
pm_file = os.path.join(data_dir, "25_Predictive_Maintenance.xlsx")
energy_file = os.path.join(data_dir, "18_Lamp_Energy_30days.xlsx")

df_pm = clean_load_excel(pm_file)
print("PM shape:", df_pm.shape)
print("PM unique Node_ID counts:\n", df_pm['Node_ID'].value_counts().head(5))

df_energy = clean_load_excel(energy_file)
print("Energy shape:", df_energy.shape)
print("Energy unique timestamps count:", df_energy['Timestamp'].nunique())
print("Energy timestamps range:", df_energy['Timestamp'].min(), "to", df_energy['Timestamp'].max())
print("Energy unique Lamp_ID counts:", df_energy['Lamp_ID'].nunique())
