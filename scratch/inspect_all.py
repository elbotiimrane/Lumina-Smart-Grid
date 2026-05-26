import os
import pandas as pd

data_dir = "data"
for f in sorted(os.listdir(data_dir)):
    if not f.endswith(".xlsx"):
        continue
    path = os.path.join(data_dir, f)
    try:
        df = pd.read_excel(path)
        print(f"=== {f} ===")
        print(f"Shape: {df.shape}")
        # Find the header row. Often files have title rows. Let's find a row that doesn't have many NaNs.
        header_row = 0
        for i in range(min(5, len(df))):
            non_nans = df.iloc[i].notna().sum()
            if non_nans > len(df.columns) * 0.5:
                header_row = i
                break
        print(f"Detected header at row {header_row}: {df.iloc[header_row].values[:8]}")
        print("-" * 20)
    except Exception as e:
        print(f"Error {f}: {e}")
