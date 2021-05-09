
import tensorflow as tf # TensorFlow
from tensorflow import keras # Keras
import numpy as np # Numpy
from tqdm import tqdm
from sklearn.model_selection import train_test_split # Easy to use function to split my data
import matplotlib.pyplot as plt # Library for Visualization
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.patches as mpatches


#import data
data_origin = pd.read_csv("CCL.csv", sep=",")
data = data_origin.copy()
data['Close/Last'] = data['Close/Last']
data['Close/Last'] = data['Close/Last'].astype('float')
data['Open'] = data['Open'].astype('float')
data['High'] = data['High'].astype('float')
data['Low'] = data['Low'].astype('float')

#get close data
noCloseData = np.array(data.drop(['Close/Last', 'Date'], 1))
closeData = np.array(data['Close/Last'])
predictionDays = 120
predictPrice = [closeData[predictionDays]]


#normalize data
scaler = MinMaxScaler(feature_range=(0,1))
normalizedData = closeData

#split into test and training data
trainingToTestRation = .66
trainDataset,testDataset = normalizedData[:int(len(normalizedData)*trainingToTestRation)], normalizedData[int(len(normalizedData)*trainingToTestRation):]

normalizedTrain = scaler.fit_transform(np.reshape(closeData,(-1,1)))
#set training data
def getTrainingData(data,predictionDays):
    xTrain = []
    yTrain = []
    for x in range(predictionDays, len(data)):
        xTrain.append(data[x-predictionDays:x,0])
        yTrain.append(data[x])
    xTrain, yTrain = np.array(xTrain), np.array(yTrain)  
    xTrain = np.reshape(xTrain,(xTrain.shape[0],xTrain.shape[1],1))
    return xTrain, yTrain

xTrain, yTrain = getTrainingData(normalizedTrain,predictionDays)
print('yTain:')
print(yTrain)

xTrain = xTrain[:,:,0]


#Linear Model
Regressor = LinearRegression()
Regressor.fit(xTrain, yTrain)

'''
#LSTM MOdel
LSTMmodel = Sequential()
LSTMmodel.add(LSTM(units = 50,return_sequences = True, input_shape=(xTrain.shape[1],1)))
LSTMmodel.add(Dropout(0.2))
LSTMmodel.add(LSTM(units = 50,return_sequences = True))
LSTMmodel.add(Dropout(0.2))
LSTMmodel.add(LSTM(units = 50))
LSTMmodel.add(Dropout(0.2))
LSTMmodel.add(Dense(units=1))

LSTMmodel.compile(optimizer='adam', loss='mean_squared_error')
LSTMmodel.fit(xTrain, yTrain, epochs=15, batch_size=32)
'''

#get input test data
def getTestData(data,predictionDays,trainingToTestRation):
    xTest = []
    totalDataset = np.concatenate((trainDataset,testDataset),axis = 0)
    testInputs = data[(len(totalDataset) - len(testDataset) - predictionDays):]
    testInputs = testInputs.reshape(-1,1)
    testInputs = scaler.transform(testInputs)
    count = 0
    for x in range(predictionDays, len(testInputs)):
        xTest.append(testInputs[x-predictionDays:x,0])
    xTest = np.array(xTest)
    xTest = np.reshape(xTest, (xTest.shape[0],xTest.shape[1],1) )
    return xTest 
testData = getTestData(closeData,predictionDays,trainingToTestRation)
testData = testData[:,:,0]

#put through model and get predictions
output = Regressor.predict(testData)
output  = scaler.inverse_transform(output)

dataset = np.concatenate((trainDataset,testDataset),axis = 0)
ActualPrice = closeData[int(len(normalizedData)*trainingToTestRation):]

#plot predictionsplt.plot(ActualPrice , color="black", label=f"Actual Price")
plt.plot(output, color="red", label=f"Predicted Price")
plt.plot(ActualPrice, color="black", label=f"Actual Price")
blackPatch = mpatches.Patch(color="red", label = "Linear Prediction")
redPatch = mpatches.Patch(color="black", label="Holding")
plt.legend(handles=[blackPatch, redPatch])
plt.xlabel("Days")
plt.ylabel("Price")
plt.title("Actual Price vs. Predicted Price using Linear Regression")
plt.show()

#shares = 10000/ActualPrice[0]
shares = 0
cash = 10000/ActualPrice[0]

print('predictionDays: ', predictionDays, ActualPrice[predictionDays + 1])

netInCash = []
notMoved = []
noMshare = 10000/ActualPrice[0]

shareDifference = []

for i in range(len(output)-1):
    if output[i+1] > ActualPrice[i]:
        if cash > 0:
            shares = cash / ActualPrice[i]
            cash = 0
    if output[i+1] < ActualPrice[i]:
        if shares > 0:
            cash = shares * ActualPrice[i]
            shares = 0
    if shares > 0:
        netInCash.append(shares*ActualPrice[i])
    else:
        netInCash.append(cash)
    notMoved.append(noMshare*ActualPrice[i])
    shareDifference.append(output[i+1] - ActualPrice[i])

plt.plot(shareDifference)
plt.show()
    
differenceInPredictions = 0
for i in range(len(output)):
    differenceInPredictions = abs(output[i]-ActualPrice[i])
    
print(differenceInPredictions)
    
plt.plot(netInCash, color="black", label="Net Cash if trading off predictions")
plt.plot(notMoved, color="red", label="Net Cash from NotTrading")
blackPatch = mpatches.Patch(color="black", label="Linear Prediction")
redPatch = mpatches.Patch(color="red", label="NotTrading")
plt.legend(handles=[blackPatch, redPatch])
plt.ylabel("Net Value")
plt.xlabel("Days Passed")
plt.title("Net Values of Shares Traded Using Linear Regression vs. Holding")
plt.show()

print("Total Shares: ", shares*ActualPrice[i])
print("Total price: ", cash)
print("Natural: ", noMshare*ActualPrice[i])