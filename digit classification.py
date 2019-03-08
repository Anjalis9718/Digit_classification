# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:27:05 2018

@author: Akshay Kumar
"""
#DO IT WITH MNIST DATA SET
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
digits = load_digits()
print(digits)
# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
print("Image Data Shape" , digits.data.shape)
# Print to show there are 1797 labels (integers from 0–9)
print("Label Data Shape", digits.target.shape)

plt.figure(figsize=(20,4))
#in enumerate is used to list up the data

for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
    
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

logistic = LogisticRegression()

logistic.fit(x_train, y_train)

logistic.predict(x_test[0].reshape(1,-1))
predictions=logistic.predict(x_test[0:10])
print(predictions)

predictions = logistic.predict(x_test)

index=0
misclassified=[]
for predict,actual in zip(predictions,y_test):
    if predict!=actual:
        misclassified.append(index)
    index+=1
    
print(misclassified)

plt.figure(figsize=(20,4))
for plotindex, wrong in enumerate(misclassified[10:15]):
    plt.subplot(1, 5, plotindex + 1)
    plt.imshow(np.reshape(x_test[wrong], (8,8)), cmap=plt.cm.gray)
    plt.title("actual: {} predicted: {}".format(y_test[wrong],predictions[wrong]), fontsize = 20)
   
# Use score method to get accuracy of model
score = logistic.score(x_test, y_test)
print(score)