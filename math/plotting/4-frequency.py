#!/usr/bin/env python3
"""
4-frequency.py
Plot a histogram of student grades.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plot a histogram of student scores for Project A.

    - X-axis: Grades (bins every 10)
    - Y-axis: Number of Students
    - Bars outlined in black
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    bins = np.arange(0, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor="black")

    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xticks(np.arange(0, 101, 10))
    plt.show()
