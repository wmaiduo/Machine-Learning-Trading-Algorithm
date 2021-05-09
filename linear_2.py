import tensorflow as tf # TensorFlow
from tensorflow import keras # Keras
import numpy as np # Numpy
from tqdm import tqdm
from sklearn.model_selection import train_test_split # Easy to use function to split my data
import matplotlib.pyplot as plt # Library for Visualization
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#import data
data_origin = pd.read_csv("Apple5YrData.csv", sep=",")
data = data_origin.copy()
data['Close/Last'] = data['Close/Last'].str.replace('$', '')
data['Open'] = data['Open'].str.replace('$', '')
data['High'] = data['High'].str.replace('$', '')
data['Low'] = data['Low'].str.replace('$', '')
data['Close/Last'] = data['Close/Last'].astype('float')
data['Open'] = data['Open'].astype('float')
data['High'] = data['High'].astype('float')
data['Low'] = data['Low'].astype('float')

#get close data
noCloseData = np.array(data.drop(['Close/Last', 'Date'], 1))
closeData = np.array(data['Close/Last'])
closeData = np.flipud(closeData)
predictionDays = [60,120,365]
predictPrice = [closeData[predictionDays]]

#normalize data
scaler = MinMaxScaler(feature_range=(0,1))
normalizedData = np.reshape(closeData,(-1,1))

#split into test and training data
normalizedTrain, normalizedTest = normalizedData[:838,:], normalizedData[838:,:]

#set training data
def getTrainingData(data,predictionDays):
    xTrain = []
    yTrain = []
    for x in range(predictionDays, len(data)):
        xTrain.append(data[x-predictionDays:x,0])
        yTrain.append(data[x,0])
    xTrain, yTrain = np.array(xTrain), np.array(yTrain)  
    xTrain = np.reshape(xTrain,(xTrain.shape[0],xTrain.shape[1],1))
    return xTrain, yTrain
xTrain60Days, yTrain60Days = getTrainingData(normalizedTrain,predictionDays[0])
xTrain120Days, yTrain120Days = getTrainingData(normalizedTrain,predictionDays[1])
xTrain365Days, yTrain365Days = getTrainingData(normalizedTrain,predictionDays[2])

xTrain60Days = xTrain60Days[:,:,0]

xTest60Days, yTest60Days = getTrainingData(normalizedTest,predictionDays[0])
xTest60Days = xTest60Days[:,:,0]

print(xTrain60Days.shape)
print(yTrain60Days.shape)


Regressor = LinearRegression()
Regressor.fit(xTrain60Days, yTrain60Days)

model_inputs = normalizedTest 
predictions = []
for inputDays in xTest60Days:
    predictions.append
    
output = Regressor.predict(xTest60Days) 
print(output)


plt.plot(normalizedTest, color="black", label=f"Actual Price")
plt.plot(predictPrice, color="red", label=f"Predicted Price")
plt.show()