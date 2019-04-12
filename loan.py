#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:26:16 2019

@author: abhaymone
"""

#importing libraries#
import numpy as np
import pandas as pd
from matplotlib  import pyplot
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from scipy.stats import normaltest
import scipy.stats as stats
import seaborn as sns
sns.set(color_codes=True)


#importing training data into pandas dataframes 
training_data = pd.read_csv("Loan_Train.csv")
training_data.head()
des = training_data.describe()
testing_data = pd.read_csv("Loan_Test.csv")


#Missing value analysis first Approach -->> drop rows with NA or imputing???. Do you impute loan_amount 
#with mean value?? Imputing categorical variable with unique category is possible. But what about numerical 
#columns ?? esp. loan_amt?
#Final thoughts -->> your testing data contains blank values . In order to predict you cannot change 
#testing data. Another important reason is I want to try and test
col_mask=training_data.isnull().any(axis=0) 
naSum = training_data.isnull().sum()   #Total NA = 149
training_data.columns
values = {'Gender': 'U', 'Married': 'U', 'Dependents': 'U', 'Self_Employed': 'U', 'Credit_History' : 2 ,'LoanAmount' : np.mean(training_data.iloc[:,8]) , 'Loan_Amount_Term' :   np.mean(training_data.iloc[:,9]) }
training_data = training_data.fillna(value=values)

training_data.head()

#encoding coloumns#
from sklearn.preprocessing import OneHotEncoder
encoder =  OneHotEncoder()
transformedCol1 = encoder.fit_transform(training_data.iloc[:,1].values.reshape(-1,1)).toarray()
transformedCol2 = encoder.fit_transform(training_data.iloc[:,2].values.reshape(-1,1)).toarray()
transformedCol3 = encoder.fit_transform(training_data.iloc[:,3].values.reshape(-1,1)).toarray()
transformedCol4 = encoder.fit_transform(training_data.iloc[:,4].values.reshape(-1,1)).toarray()
transformedCol5 = encoder.fit_transform(training_data.iloc[:,5].values.reshape(-1,1)).toarray()
transformedCol10 = encoder.fit_transform(training_data.iloc[:,10].values.reshape(-1,1)).toarray()
transformedCol11 = encoder.fit_transform(training_data.iloc[:,11].values.reshape(-1,1)).toarray()
transformed_training_data = np.c_[transformedCol1[:,[0,1]],transformedCol2[:,[0,1]],transformedCol3[:,[0,1,2,3]],transformedCol4[:,0],transformedCol5[:,[0,2]],transformedCol10[:,[0,1]],transformedCol11[:,[0,1,2]],training_data.iloc[:,[6,7,8,9]]]
#Spliting data into features and label
x_train = transformed_training_data #shape(614,20)

#Check n/a for each column
checkNA = np.isnan(x_train)

#Convert lable values into numbers
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y_train = lb.fit_transform(training_data.iloc[:,12].values)


#data analysis
sns.distplot(x_train[:,16],bins=100,hist=False, rug= True)

sns.dis















