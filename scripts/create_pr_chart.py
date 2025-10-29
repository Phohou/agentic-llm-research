import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import pyarrow.parquet as pq

def loadDataFrame():
    """Loads the dataset from the parquet file."""
    dataPath = Path("output")

    if not dataPath.exists():
        print(f"File {dataPath} not found.")
        return None
    
    print(f"Loading data from {dataPath}")
    df = pd.read_parquet(dataPath)
    
    # Checking the structure of the DataFrame
    print(f"Loaded {len(df)} issues")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:\n{df.head()}")

    return df