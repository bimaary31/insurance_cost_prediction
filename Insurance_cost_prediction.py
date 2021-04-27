# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:50:52 2021

@author: ARY - PC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#Loading data insurance
insurance_data = pd.read_csv(r'D:/insurance/insurance.csv')
print(insurance_data.head(20))

print(insurance_data.shape)
print(insurance_data.info())

df= insurance_data.isna().sum()
print('Jumlah Missing Value:\n', df)

print('statistic:\n', insurance_data.describe())


#Exploration Data Age Distribution
sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance_data['age'])
plt.title("Age Distribution")
plt.show()

#Exploration data Gender Distribution
plt.figure(figsize=(6,6))
sns.countplot(x='sex', hue='region', data=insurance_data)
plt.title('Sex Distribution')
plt.show()

print('jumlah data masing_gender:\n', insurance_data['sex'].value_counts())

#Exploration Distribution BMI
plt.figure(figsize=(6,6))
sns.displot(insurance_data['bmi'])
plt.title("BMI Distribution")
plt.show() 
print('normal bmi --> 18.5-24.9')

#Exploration Children Distribution
plt.figure(figsize=(6,6))
sns.countplot(x='children',hue='region', data=insurance_data)
plt.title('Children Distribution')
plt.show()

#Exploration Smoker distribution 
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', hue='region', data=insurance_data)
plt.title('Smoker Distribution')
plt.show()

print("Jumlah Perokok dan Bukan:\n", insurance_data['smoker'].value_counts())


#Data Preprocessing Encoding
insurance_data.replace({'sex':{'male':0, 'female':1}}, inplace=True)
insurance_data.replace({'smoker':{'yes':0, 'no':1}},inplace=True)
insurance_data.replace({'region':{'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}},inplace=True)

X = insurance_data.drop(columns ='charges', axis=1)
Y = insurance_data['charges']

print(X)
print(Y)

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

print('Jumlah baris dan kolom dari x_train adalah:', x_train.shape,', sedangkan Jumlah baris dan kolom dari y_train adalah:', y_train.shape)
print('Prosentase Churn di data Training adalah:')
print(y_train.value_counts(normalize=True))
print('Jumlah baris dan kolom dari x_test adalah:', x_test.shape,', sedangkan Jumlah baris dan kolom dari y_test adalah:', y_test.shape)
print('Prosentase Churn di data Testing adalah:')
print(y_test.value_counts(normalize=True))
print(X.shape, x_train.shape, x_test.shape)

#Model Training
#Linear Regression

regressor= LinearRegression()
regressor.fit(x_train, y_train)

'MODEL EVALUATION'
#predict training
training_data_prediction = regressor.predict(x_train)

#R Squared
R2_train = metrics.r2_score(y_train, training_data_prediction)
print('R square Values :', R2_train)

#predict test
test_data_prediction = regressor.predict(x_test)
#R Squared
r2_test =metrics.r2_score(y_test, test_data_prediction)
print('R Squared Values:', r2_test)

'''Building Data Predict'''

input_data= (19,1,27.9,0,0,1)

#Changing input data to  numpy array
input_data_as_numpy_array=  np.asarray(input_data)

#reshape 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print(prediction)
print('Insurance Cost in USD', prediction[0])
