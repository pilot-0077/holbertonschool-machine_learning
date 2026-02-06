#!/usr/bin/env python3
"""
0-line.py
Plot y as a red line graph.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plot y = x^3 as a solid red line.

    The x-axis ranges from 0 to 10.
    """
    y = np.arange(0, 11) ** 3
    x = np.arange(0, 11)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, "r-")
    plt.xlim(0, 10)
    plt.show()
