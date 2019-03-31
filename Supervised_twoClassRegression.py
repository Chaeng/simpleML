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

#plot data

    X, y = mglearn.datasets.make_wave(n_samples=40)
    plt.plot(X, y, 'o')
    plt.ylim(-3, 3)
    plt.xlabel("Feature")
    plt.ylabel("Target")