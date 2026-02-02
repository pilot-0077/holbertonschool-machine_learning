1. Scatter
Complete the following source code to plot x ↦ y as a scatter plot:

The x-axis should be labeled Height (in)
The y-axis should be labeled Weight (lbs)
The title should be Men's Height vs Weight
The data should be plotted as magenta points
hbt-ml@Holberton-ML:~$ cat 1-scatter.py
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def scatter():

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    # your code here
Result

hbt-ml@Holberton-ML:~$ cat 1-main.py
#!/usr/bin/env python3

scatter = __import__('1-scatter').scatter

scatter()
hbt-ml@Holberton-ML:~$ ./1-main.py
Example plot

Repo:

GitHub repository: holbertonschool-machine_learning
Directory: math/plotting
File: 1-scatter.py
