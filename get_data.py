import csv
from tkinter import *
import numpy as np
from sklearn import linear_model 
import matplotlib.pyplot as plt

dates = []
prices = []
def get_data(filename):
    with open(filename,'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        for row in csvFileReader:
            dates.append(int(row[0]))
            prices.append(float(row[1]))
    return         

def predict_price(dates,prices,x):
    linear_mod = linear_model.LinearRegression()
    dates = np.reshape(dates,(len(dates),1))
    prices = np.reshape(prices,(len(prices),1))
    linear_mod.fit(dates,prices)
    predicted_price = linear_mod.predict(x)
    return predicted_price[0][0]


def show_plot(dates,prices):
    linear_mod = linear_model.LinearRegression()
    dates = np.reshape(dates,(len(dates),1))
    prices = np.reshape(prices,(len(prices),1))
    linear_mod.fit(dates,prices)
    plt.scatter(dates,prices,color='yellow')
    plt.plot(dates,linear_mod.predict(dates),color='blue',linewidth=3)
    plt.show()
    return

get_data('google.csv')

show_plot(dates,prices)

x=[[130]]

predicted_price=predict_price(dates,prices,x)
print("Predicted price",x[0][0],"th day is",predicted_price)
root = Tk()
T = Text(root,height=18,width=40)
root.title("Prediction")
T.pack()
T.insert(END,"The stock price for",x[0][0],"th div is "+str(predicted_price))
mainloop()
