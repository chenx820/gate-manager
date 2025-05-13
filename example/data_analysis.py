from conductorquantum import ConductorQuantum

# Initialize client with API key
client = ConductorQuantum(token="your_api_key_here")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

parent_dir = "20250406/data/"
file_name = f"20250406_CT_[b_D]_vs_[b_P3]_[t_s&b_S]_diamond_run1.txt"


df = pd.read_csv(
    parent_dir + file_name, sep=r"\s+", skiprows=1, names=["Y", "X", "Z"]
)  # Assign column names