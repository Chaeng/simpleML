# 
# Simple python project for classifying Iris flower: 1/2
# Credit: Introduction to Machine Learning with Python by Andreas C. Muller & Sarah Guido

# Purpose: To study Machine Learning concepts and applications. The original code was created
#          by Muller & Guido and modified by the author for educational purpose only.
# Author: Ploypaphat Saltz
# Date: 03/30/2019
#

import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from Supervised_twoClassClassification import plt
#import matplotlib.pyplot as plt
if __name__ == "__main__":
    
    def printAccuracy(classifier, X_test, y_test):
        print("Test accuracy: {:.2f}".format(classifier.score(X_test, y_test)))
    
    def plotBreastCancer():
        
        cancer = load_breast_cancer()
        print("cancer.keys():\n", cancer.keys())
    
    def plotBoston():
        
        boston = load_boston()
        print("Datashap:", boston.data.shape)
        X, y = mglearn.datasets.load_extended_boston()
        print("X.shape:", X.shape)
        mglearn.plots.plot_knn_classification(n_neighbors=2)
    
    def k_nearest():
        #split data =>[training set 75%, test set25%]
        
        X, y = mglearn.datasets.make_forge()
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
        
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X_train, y_train)
        
        # make prediction on test data
        #print("Test set prediction:", clf.predict(X_test))
        #printAccuracy(clf, X_test, y_test)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        
        for n_neighbors, ax in zip([1, 3, 9], axes):
            clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
            mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
            mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
            ax.set_title("{} neighbor(s)".format(n_neighbors))
            ax.set_xlabel("feature 0")
            ax.set_ylabel("feature 1")
        axes[0].legend(loc=3)
        
    k_nearest()