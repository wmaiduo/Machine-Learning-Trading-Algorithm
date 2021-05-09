import torch # PyTorch
import tensorflow as tf # TensorFlow
from tensorflow import keras # Keras
import numpy as np # Numpy
from tqdm import tqdm
from sklearn.model_selection import train_test_split # Easy to use function to split my data
import matplotlib.pyplot as plt # Library for Visualization
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import datetime as dt

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
predictionDays = [60,120,365]
predictPrice = [closeData[predictionDays]]


#normalize data
scaler = MinMaxScaler(feature_range=(0,1))
normalizedData = scaler.fit_transform(np.reshape(closeData,(-1,1)))

#split into test and training data
trainingToTestRation = .66
normalizedTrain, normalizedTest = normalizedData[:int(len(normalizedData)*trainingToTestRation),:], normalizedData[int(len(normalizedData)*trainingToTestRation):,:]

#set training data
def getTrainingData(data,predictionDays):
    xTrain = []
    yTrain = []
    for x in range(predictionDays, len(data)):
        xTrain.append(data[x-predictionDays:x,0])
        yTrain.append(data[x,0])
    xTrain, yTrain = np.array(xTrain), np.array(yTrain)  
    #xTrain = np.reshape(xTrain,(xTrain.shape[0],xTrain.shape[1],1))
    return xTrain, yTrain
xTrain60Days, yTrain60Days = getTrainingData(normalizedTrain,predictionDays[0])
xTrain120Days, yTrain120Days = getTrainingData(normalizedTrain,predictionDays[1])
xTrain365Days, yTrain365Days = getTrainingData(normalizedTrain,predictionDays[2])


linear = linear_model.LinearRegression()
linear.fit(xTrain60Days,yTrain60Days)

'''
LSTMmodel = Sequential()
LSTMmodel.add(LSTM(units = 50,return_sequences = True, input_shape=(xTrain60Days.shape[1],1)))
LSTMmodel.add(Dropout(0.2))
LSTMmodel.add(LSTM(units = 50,return_sequences = True))
LSTMmodel.add(Dropout(0.2))
LSTMmodel.add(LSTM(units = 50))
LSTMmodel.add(Dropout(0.2))
LSTMmodel.add(Dense(units=1))

LSTMmodel.compile(optimizer='adam', loss='mean_squared_error')
LSTMmodel.fit(xTrain60Days, xTrain60Days, epochs=15, batch_size=32)
'''

testInputs = closeData[len(closeData) - len(normalizedTest) - predictionDays[0]:]
testInputs = testInputs.reshape(-1,1)
testInputs = scaler.transform(testInputs)

predictions = []
for x in range(predictionDays[0], len(testInputs)):
    predictions.append(testInputs[x-predictionDays[0]:x,0])
    
predictions = np.array(predictions)
#predictions = np.reshape(predictions, (predictions.shape[0],predictions.shape[1],1) )
#predictPrice = LSTMmodel.predict(predictions)
predictPrice = linear.predict(predictions)
predictPrice = scaler.inverse_transform(predictPrice)

predictPrice = np.flip(predictPrice,0)
actualPrice = np.flip(closeData[len(normalizedTrain):],0)

'''
#get input test data
def getTestData(data,predictionDays,trainingToTestRation):
    testData = []
    testInputs = data[int(len(data) - len(data)*(1-trainingToTestRation) - predictionDays):]
    testInputs = testInputs.reshape(-1,1)
    testInputs = scaler.transform(testInputs)
    for x in range(predictionDays, int(len(data)*(1-trainingToTestRation))):
        testData.append(testInputs[x-predictionDays:x,0])
    testData = np.array(testData)
    testData = np.reshape(testData, (testData.shape[0],testData.shape[1],1) )
    return testData

testData = getTestData(closeData,predictionDays[0],trainingToTestRation)

#put through model and get predictionc
predictions = LSTMmodel.predict(testData)
predictions = scaler.inverse_transform(predictions)
predictions = np.flip(predictions,0)
ActualPrice = np.flip(closeData[int(len(closeData)*trainingToTestRation):],0)
'''


#plot predictionc
plt.plot(ActualPrice , color="black", label=f"Actual Price")
plt.plot(predictPrice, color="red", label=f"Predicted Price")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()


shares = 10000/actualPrice[predictionDays + 1]
cash = 0

print('predictionDays: ', predictionDays, actualPrice[predictionDays + 1])

netInCash = []
notMoved = []
noMshare = 10000/actualPrice[predictionDays + 1]

for i in range(predictionDays + 1, len(closeData) - predictionDays - 2):
    if predictPrice[i+1] > actualPrice[i]:
        if cash > 0:
            shares = cash / actualPrice[i]
            cash = 0
    if predictPrice[i+1] < actualPrice[i]:
        if shares > 0:
            cash = shares * actualPrice[i]
            shares = 0
    print(i)
    print("shares: ", shares)
    print("cash: ", cash)
    if shares > 0:
        netInCash.append(shares*actualPrice[i])
    else:
        netInCash.append(cash)
    notMoved.append(noMshare*actualPrice[i])

pyplot.plot(netInCash, color="black", label="Net Shares in Cash if Traded According to Linear Regression")
pyplot.plot(notMoved, color="red", label="NetShares in Cash if not traded")
blackPatch = mpatches.Patch(color="black", label="Linear Regression")
redPatch = mpatches.Patch(color="red", label="Not Traded")
pyplot.legend(handles=[blackPatch, redPatch])
pyplot.ylabel("Net Value")
pyplot.xlabel("Days Passed")
pyplot.title("Net Values of Shares Traded Using Linear Regression vs. Untraded Shares")
pyplot.show()

print("Total Shares: ", shares*actualPrice[i])
print("Total price: ", cash)
print("Natural: ", noMshare*actualPrice[i])