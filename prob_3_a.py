# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 11:32:11 2019

@author: nipun
"""

"""
PROBLEM 3, PART A

"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score



data = pd.read_csv("poly_data.csv", sep = ' ') ##column 1 is X and column 2 is Y


X = data.iloc[:, 0].values
y = data.iloc[:, 1].values

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

num_degrees = 20 #starting from 1 to 20; so we consider 20 models in this case

k = 8  #the value of k. Found this value by considering all values between 5 and 10 (and running multiple iterations for each) and choosing the one which gave lowest error.

test_error = np.zeros(num_degrees)

for degrees in range(0, num_degrees):
    polynomial_features = PolynomialFeatures(degree = degrees+1)
    x_poly = polynomial_features.fit_transform(X)
    
    model = LinearRegression() ##Getting the model
    model.fit(x_poly, y)
    
    scores = np.zeros(k)
    scores = cross_val_score(model, x_poly, y, cv=k, scoring= 'neg_mean_squared_error')  ##Calculating the MSE using Cross-validation
    scores = -scores
    test_error[degrees] = np.mean(scores)
    print("Cross Validation Error for Degree", degrees+1, "is", test_error[degrees])

plt.plot(np.arange(1, 21, 1), test_error, color = 'blue') 
  
plt.title('Polynomial Regression for Degrees from 2-20') 
plt.xlabel('Degree') 
plt.ylabel('MSE') 
  
plt.show()