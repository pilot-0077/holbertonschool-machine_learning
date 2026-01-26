#!/usr/bin/env python3
"""Create a pandas DataFrame from a dictionary and store it in df."""

import pandas as pd

data = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"],
}

index = ["A", "B", "C", "D"]

df = pd.DataFrame(data, index=index)
