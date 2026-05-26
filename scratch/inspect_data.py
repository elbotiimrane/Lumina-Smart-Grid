import pandas as pd
import os

data_dir = "data"
lamp_energy_file = os.path.join(data_dir, "18_Lamp_Energy_30days.xlsx")
df = pd.read_excel(lamp_energy_file, nrows=100)
print("Lamp Energy Columns:", df.columns)
print("Unique times in first 100 rows:", df['Timestamp'].unique())
print("Status unique values:", df['Status'].unique() if 'Status' in df.columns else 'N/A')

pm_file = os.path.join(data_dir, "25_Predictive_Maintenance.xlsx")
df_pm = pd.read_excel(pm_file)
print("PM Columns:", df_pm.columns)
print("PM Sample:")
print(df_pm.head(3))
