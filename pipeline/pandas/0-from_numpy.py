#!/usr/bin/env python3
import pandas as pd

def from_numpy(array):
    cols = [chr(ord('A') + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=cols)
