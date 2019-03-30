from sklearn.datasets import load_iris


iris_dataset = load_iris()

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
print("Data: " , type(iris_dataset['data']))
print()
