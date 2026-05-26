import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from prediction.graph_builder import clean_load_excel

kpi_file = os.path.join("data", "23_Daily_KPI_Summary.xlsx")
df_clean = clean_load_excel(kpi_file)
print("Cleaned Columns:", df_clean.columns.tolist())
print("Cleaned Shape:", df_clean.shape)
print("Cleaned Head:")
print(df_clean.head(2))
