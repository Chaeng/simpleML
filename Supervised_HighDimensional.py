# 
# Simple python project for classifying Iris flower: 1/2
# Credit: Introduction to Machine Learning with Python by Andreas C. Muller & Sarah Guido

# Purpose: To study Machine Learning concepts and applications. The original code was created
#          by Muller & Guido and modified by the author for educational purpose only.
# Author: Ploypaphat Saltz
# Date: 03/30/2019
#

#import mglearn
#import matplotlib.pyplot as plt
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    print("cancer.keys():\n", cancer.keys())