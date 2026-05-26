import pandas as pd
df = pd.read_excel("data/23_Daily_KPI_Summary.xlsx")
for i in range(10):
    print(f"Row {i}:", df.iloc[i].values.tolist())
