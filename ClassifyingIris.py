# 
# Simple python project for classifying Iris flower: 1/2
# Credit: Introduction to Machine Learning with Python by Andreas C. Muller & Sarah Guido

# Purpose: To study Machine Learning concepts and applications. The original code was created
#          by Muller & Guido and modified by the author for educational purpose only.
# Author: Ploypaphat Saltz
# Date: 03/30/2019
#
import pandas as pd

from sklearn.datasets import load_iris

iris_dataset = load_iris()

if __name__ == "__main__":

## print keys
# print("Key of iris_dataset:\n", iris_dataset.keys())
# output: dict_keys(['DESCR', 'data', 'target_names', 'feature_names', 'target'])

## print value:
##     print values of 'DESCR' => short description of the dataset
    print(iris_dataset['DESCR'][:193] + "\n...")

##     print values of 'target_names' => array of spicies of flower
    print("Target names: " , iris_dataset['target_names'])
    print()

##     print values of 'feature_names' => array of description of each feature
    print("Feature names: " , iris_dataset['feature_names'])
    print()

##     print values of 'feature_names' => 150x4 array of the measurement
##                                        150 flowers and its measurement
##                                        [:5] is printing the first 5 rows
    print("Data: " , iris_dataset['data'][:5])
    print()

##     print the shape of array
    print("Shape of data: " , iris_dataset['data'].shape)
    print()

##     print the target => each type encoded to 0-2 (3 kinds of iris)
    print("Target:\n" , iris_dataset['target'])
    print()

##     print the target name
    print("Target names:\n" , iris_dataset['target_names'])
    print()