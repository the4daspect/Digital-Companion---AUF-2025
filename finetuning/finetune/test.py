import os
from pathlib import Path
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)


home_dir = parent_dir = Path(__file__).resolve().parent.parent
data_dir = parent_dir / "data"
print(data_dir)

for entry in data_dir.iterdir():
    print(entry.name)

df = pd.read_csv(data_dir / "humanandllm.csv")
print(df)