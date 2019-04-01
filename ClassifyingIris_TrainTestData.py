# 
# Simple python project for classifying Iris flower: 2/2
# Credit: Introduction to Machine Learning with Python by Andreas C. Muller & Sarah Guido

# Purpose: To study Machine Learning concepts and applications. The original code was created
#          by Muller & Guido and modified by the author for educational purpose only.
# Author: Ploypaphat Saltz
# Date: 03/30/2019
#

import mglearn
from ClassifyingIris import iris_dataset, pd
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)


iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

if __name__ == "__main__":
    pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15),       marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)
