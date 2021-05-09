import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import matplotlib.patches as mpatches
import pickle

data_origin = pd.read_csv("BA.csv", sep=",")
data = data_origin.copy()

noCloseData = np.array(data.drop(['Close/Last', 'Date'], 1))
closeData = np.array(data['Close/Last'])

predictionDays = 60
predictPrice = []

#train model for every sixty passed days and use the current day data to predict next day's stock price
for i in range(0, len(closeData) - predictionDays - 1):
    x_train = noCloseData[i:i + predictionDays]
    y_train = closeData[i + 1:i + predictionDays + 1]
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    x_test = noCloseData[i + predictionDays + 1].reshape(1, -1)
    prediction = linear.predict(x_test)
    predictPrice.append(prediction)

actualPrice = closeData[predictionDays + 1: len(closeData)]

#plot
pyplot.plot(actualPrice, color="black", label=f"Actual Price")
pyplot.plot(predictPrice, color="red", label=f"Predict Price")
blackPatch = mpatches.Patch(color="black", label="Actual Price")
redPatch = mpatches.Patch(color="red", label="Predicted Price")
pyplot.legend(handles=[blackPatch, redPatch])
pyplot.ylabel("Closing Price")
pyplot.xlabel("Days Passed")
pyplot.show()

shares = 10000/actualPrice[predictionDays + 1]
cash = 0

print('predictionDays: ', predictionDays, actualPrice[predictionDays + 1])

netInCash = []
notMoved = []
noMshare = 10000/actualPrice[predictionDays + 1]

#for i in range(predictionDays, predictionDays + 2):
for i in range(predictionDays + 1, len(closeData) - predictionDays - 2):
    if predictPrice[i] > actualPrice[i]:
        if cash > 0:
            shares = cash / actualPrice[i]
            cash = 0
    if predictPrice[i] < actualPrice[i]:
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

pyplot.plot(netInCash, color="black")
pyplot.plot(notMoved, color="red")
pyplot.show()

print(cash)
print(shares)




