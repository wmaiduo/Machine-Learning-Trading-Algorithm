import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np



data_origin = pd.read_csv("Apple5YrData.csv", sep=",")
data = data_origin.copy()

print('raw data of Apple')
print(data.head())

data['Close/Last'] = data['Close/Last'].str.replace('$', '')
data['Open'] = data['Open'].str.replace('$', '')
data['High'] = data['High'].str.replace('$', '')
data['Low'] = data['Low'].str.replace('$', '')
data['Close/Last'] = data['Close/Last'].astype('float')
data['Open'] = data['Open'].astype('float')
data['High'] = data['High'].astype('float')
data['Low'] = data['Low'].astype('float')

data = data.drop(['Close/Last', 'Date'], 1)

data_X = data.loc[:,data.columns !=  'Close/Last' ]
data_Y = data['Close/Last']

train_X, test_X, train_y,test_y = train_test_split(data_X,data_Y,test_size=0.25)

print('\n\nTraining Set')
print(train_X.head())
print(train_Y.head())