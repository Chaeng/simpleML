# 
# Simple python project for classifying Iris flower: 1/2
# Credit: Introduction to Machine Learning with Python by Andreas C. Muller & Sarah Guido

# Purpose: To study Machine Learning concepts and applications. The original code was created
#          by Muller & Guido and modified by the author for educational purpose only.
# Author: Ploypaphat Saltz
# Date: 03/30/2019
#

from ClassifyingIris import iris_dataset
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from ClassifyingIris_TrainTestData import X_train, X_test, y_train, y_test

#if __name__ == "__main__":
    
# knn encapsulates algorithm to build model from the training data
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)

# test predicting
prediction = knn.predict(X_new)

print("\n///////////////////////////////////////////////////////////////////")

print("Prediction:", prediction)
print("Predicted target name:", iris_dataset['target_names'][prediction])

print("\n///////////////////////////////////////////////////////////////////")

#Evaluating the model
#   To evaluate, predict each iris in the test data (25%),
#   then compare it against its label. Then compute the 'accuracy'

y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)

print("\n///////////////////////////////////////////////////////////////////")

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

print("\n///////////////////////////////////////////////////////////////////")