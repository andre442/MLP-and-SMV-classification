# MLP-and-SMV-classification-
Simple comparative of SVM and MLP classifiers using Heart Disease UCI Database. This project shows in a simple way how to use the sklearn framework to generate a classifier and how to measure the results based on the confusion matrix and the accuracy.


The features of the dataset

![features](https://user-images.githubusercontent.com/50015049/57184286-e1d93d00-6e8f-11e9-922b-1a0e612b00ca.png)


After using StandardScaler from sklearn library we can slip in test and train data. In this case i´ve used test_size=0.3
Now using SVM model with this parameters -> (kernel = 'rbf', random_state = 0, C = 2.0)
The confusion the result matrix is:

![smv](https://user-images.githubusercontent.com/50015049/57184282-d84fd500-6e8f-11e9-936c-2edc9ad40335.png)

and the accuracy obtained was 0.824

Using the multi layer perceptron NN, with -> (max_iter=1000, tol=0.000010)
The confusion the result matrix is:

![MPL](https://user-images.githubusercontent.com/50015049/57184281-cb32e600-6e8f-11e9-9bc9-4b3fce570407.png)

and the accuracy obtained was 0.868

Note that the division between the training and testing database used in the classifiers was the same.
In this repository the complete code and the dataset in .csv format are available.

