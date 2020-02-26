# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:31:14 2019

@author: nipun
"""

"""
PROBLEM 3, PART B

"""



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


data = pd.read_csv("poly_data.csv", sep = ' ') ##column 1 is X and column 2 is Y


X = data.iloc[:, 0:1].values ##Due to this, X is now a 2d vector. So, there's no need to use the reshape command.
y = data.iloc[:, 1].values



polynomial_features = PolynomialFeatures(degree = 3) ##Degree = 3 gave us the least error.
x_poly = polynomial_features.fit_transform(X)

model = LinearRegression() ##Getting the model
model.fit(x_poly, y)

coeff = model.coef_
print("The coefficients are:", coeff)

intercept = model.intercept_
print("The intercept is:", intercept)


y_hat = model.predict(x_poly)


plt.figure()
plt.scatter(X, y, color = 'lightgreen')  ##To plot the dataset
  
plt.scatter(X, y_hat, color = 'violet')  ##To plot yhat
plt.title('Polynomial Regression') 
plt.xlabel('X') 
plt.ylabel('Y') 
  
plt.show() 

