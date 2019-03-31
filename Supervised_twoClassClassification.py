# 
# Simple python project for classifying Iris flower: 1/2
# Credit: Introduction to Machine Learning with Python by Andreas C. Muller & Sarah Guido

# Purpose: To study Machine Learning concepts and applications. The original code was created
#          by Muller & Guido and modified by the author for educational purpose only.
# Author: Ploypaphat Saltz
# Date: 03/30/2019
#

import mglearn
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X, y = mglearn.datasets.make_forge()

#plot data
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(["Class 0", "Class 1"], loc=4)
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    print("X.shape:", X.shape)
