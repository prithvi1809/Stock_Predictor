#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,LSTM


import yfinance as yf


# In[2]:


#training_data_len=1610
#scaled_data=(2012,1)
#x_train=(1550,60,1)
#y_train=(1550,)


# In[3]:


#downloading data from internet of apple stock
df=yf.download('NVDA',start='2012-01-01',end='2020-01-01')
plt.figure(figsize=(16,8))
plt.plot(df['Close'])
plt.show()


# In[4]:


df.shape


# In[5]:


#create a new data with only close column
data=df.filter(['Close'])
#convert to numpy array
dataset=data.values


# In[6]:


#1610 out of 2012 for training.ie 80 percent.
training_data_len=math.ceil(0.8*data.shape[0])
training_data_len=int(training_data_len)
print(training_data_len)


# In[7]:


data


# In[8]:


#scale the data
sc=MinMaxScaler(feature_range=(0,1))
scaled_data=sc.fit_transform(data)
print(scaled_data.shape)


# In[9]:


#create the training data set
train_data=scaled_data[0:training_data_len,:]
#split the train_data into x_train and y_train data sets.
x_train=[]
y_train=[]

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])  #for x in 0 to 59 ,y is 60 ...for x in 1 to 60 y is 61....
    y_train.append(train_data[i,0])


# In[10]:


#convert into arrays
x_train,y_train=np.array(x_train),np.array(y_train)
#shape of x_train is (1550,60)
#reshape the data as LSTM accepts 3D data whereas our data is 2D 
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


# In[11]:


#Build The LSTM model
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
#Compile the model
model.compile(optimizer='adam',loss='mean_squared_error')


# In[12]:


#Train the model
model.fit(x_train,y_train,batch_size=1,epochs=3)


# In[13]:


#Create a testing data set
#create a new array containing scaled value from index 1550 to 2012
test_data=scaled_data[training_data_len-60:,:]
#create data set x_test and y_test
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])


# In[14]:


x_test=np.array(x_test)
#Reshape the data
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
x_test.shape


# In[146]:


#Get the models predicted price value
predictions=model.predict(x_test)
predictions=sc.inverse_transform(predictions)
print(predictions.shape)


# In[16]:


#Get the root mean square error
rmse=np.sqrt(np.mean(predictions-y_test)**2)
print(rmse)


# In[17]:


#Plot the data
train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Prediction'],loc='lower right')


# In[57]:


#assembling data for prediction in next one year
future_test=[]
for i in range(1):
    future_test.append(test_data[len(test_data)-60:len(test_data),0])
future_test=np.array(future_test)   
future_test=np.reshape(future_test,(future_test.shape[0],future_test.shape[1],1))
print(future_test.shape)
#prediction of future stock price
future_pred=model.predict(future_test)
future_pred=sc.inverse_transform(future_pred)
print(future_pred)


# In[147]:


#Future prediction of stocks with using the data of only the blue colour on the above plot 
#ie with data upto mid 2018 ..i predicted data upto year 2020.
#future_x_train is my final future x_test matrix that has shape 402*60*1 ....ie data to predict the stock price
future_x_train=[]
xx_test=[]
yy_pred=[]
xxx_test=[]
xx_test=np.array(xx_test)
xxx_test=np.array(xxx_test)
for i in range(len(test_data)-60):
    xx_test=[]
    for j in range(1):
        if i<60:
            xx_test=np.append(xx_test,scaled_data[training_data_len-60+i:training_data_len,0]) 
        else:
            xx_test=np.append(xx_test,yy_pred[i-60:i])

    if i<60:
        xxx_test=np.append(xx_test,np.ravel(yy_pred))
        xx_test=xxx_test
    else:
        xxx_test=xx_test
    xx_test=np.reshape(xx_test,(1,xx_test.shape[0],1))
    print(xx_test.shape)
    pred=model.predict(xx_test)
    yy_pred.append(pred)
    print(xxx_test)
    print(pred)
    print(i)
    future_x_train.append(xxx_test)


# In[148]:


future_x_train=np.array(future_x_train)
future_x_train=np.reshape(future_x_train,(future_x_train.shape[0],future_x_train.shape[1],1))
future_x_train.shape


# In[149]:


future_predictions=model.predict(future_x_train)
print(future_predictions.shape)
future_predictions=sc.inverse_transform(future_predictions)


# In[151]:


#Plot the data
train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions
valid['Future_predictions']=future_predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions','Future_predictions']])
plt.legend(['Train','Val','Prediction','Future_predictions'],loc='lower right')


# In[ ]:




