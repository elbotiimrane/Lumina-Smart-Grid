import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from prediction.graph_builder import clean_load_excel

print("Loading Lamp Energy...")
df_energy = clean_load_excel("data/18_Lamp_Energy_30days.xlsx")
print("Energy columns:", df_energy.columns.tolist())
print("Energy shape:", df_energy.shape)
print("Unique lamps:", df_energy["Lamp_ID"].nunique())
print("Energy sample:\n", df_energy.head(3))

print("\nLoading Predictive Maintenance...")
df_pm = clean_load_excel("data/25_Predictive_Maintenance.xlsx")
print("PM columns:", df_pm.columns.tolist())
print("PM shape:", df_pm.shape)
print("PM sample:\n", df_pm.head(3))
