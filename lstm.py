import torch # PyTorch
import tensorflow as tf # TensorFlow
from tensorflow import keras # Keras
import numpy as np # Numpy
from tqdm import tqdm
from sklearn.model_selection import train_test_split # Easy to use function to split my data
import matplotlib.pyplot as plt # Library for Visualization
from torchvision import datasets, transforms
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import datetime as dt
import pandas_datareader as web

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

xTest60Days, yTest60Days = getTrainingData(normalizedTest,predictionDays[0])
print(xTrain60Days.shape[1])
LSTMmodel = Sequential()

LSTMmodel.add(LSTM(units = 50,return_sequences = True, input_shape=(xTrain60Days.shape[1],1)))
LSTMmodel.add(Dropout(0.2))
LSTMmodel.add(LSTM(units = 50,return_sequences = True))
LSTMmodel.add(Dropout(0.2))
LSTMmodel.add(LSTM(units = 50))
LSTMmodel.add(Dropout(0.2))
LSTMmodel.add(Dense(units=1))

LSTMmodel.compile(optimizer='adam', loss='mean_squared_error')
LSTMmodel.fit(xTrain60Days, yTrain60Days, epochs=15, batch_size=32)




model_inputs = normalizedTest 
predictions = []
for inputDays in xTest60Days:
    predictions.append
    
output = LSTMmodel.predict(xTest60Days) 
print(output)
xTest = []
#for x in range(predictionDays[0], len(xTest60Days)):
    #xTest.append(LSTMmodel.predict(xTest60Days[x]))
#xTest = np.array(xTest)
#plot
plt.plot(normalizedTest, color="black", label=f"Actual Price")
plt.plot(predictPrice, color="red", label=f"Predicted Price")
plt.show()