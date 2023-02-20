#%% import libraries

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score

from keras import Sequential
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.layers import Dense,Dropout,LSTM

train_path = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
test_path = os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')

#%% 

# data loading
df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)


# EDA
df.info()
df.isna().sum()

# data cleaning
df['cases_new'] = pd.to_numeric(df['cases_new'],errors='coerce') #tukarkan object kepada na value
# df['date'] = pd.to_datetime(df['date'])

# convert NA to values
df['cases_new'] = df['cases_new'].interpolate(method='polynomial',order=2)
df_test['cases_new'] = df_test['cases_new'].interpolate(method='polynomial',order=2)

plt.figure()
plt.plot(df['cases_new'])
plt.show()

# feature selection
data = df['cases_new'].values

#%% 

# Model development
mms = MinMaxScaler()
data = mms.fit_transform(np.expand_dims(data,axis=-1))

x_train = []
y_train = []
win_size = 30

for i in range(win_size,len(data)):
    x_train.append(data[i-win_size:i])
    y_train.append(data[i])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, train_size=0.7, random_state=123)

# %%
input_shape = np.shape(x_train)[1:]

model = Sequential()
model.add(LSTM(64,input_shape=input_shape,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(1,activation='relu'))
model.compile(optimizer='adam', loss='mse', metrics='mse')
model.summary()
plot_model(model, show_shapes=True)

log_dir = os.path.join(os.getcwd(),'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = TensorBoard(log_dir=log_dir)
hist = model.fit(x_train,y_train,batch_size=32, epochs=700,callbacks=[tb_callback])

# %% prediction test 
y_true = y_test
y_pred = model.predict(x_test)

print(mean_absolute_error(y_true,y_pred))
print('MAPE error is {}'.format(mean_absolute_percentage_error(y_true,y_pred)))
print(r2_score(y_true,y_pred))

#%% model analysis
print(hist.history.keys())

plt.figure()
plt.plot(hist.history['mse'])
# plt.plot(hist.history['val_mse'])
# plt.legend(['mse', 'validation mse'])
plt.show()

# %% 

# compile train and test data
data_tot = pd.concat((df,df_test))
data_tot_target = data_tot['cases_new'].values 
data_tot_target = mms.transform(np.expand_dims(data_tot_target,axis=-1))

x_actual = []
y_actual = []

for i in range(len(df),len(data_tot_target)): # get x and y from combine data 
    x_actual.append(data_tot_target[i-win_size:i])
    y_actual.append(data_tot_target[i])

x_actual = np.array(x_actual)
y_actual = np.array(y_actual)

# prediction
y_pred = model.predict(x_actual)

plt.figure()
plt.plot(y_pred,color='red')
plt.plot(y_actual,color='blue')
plt.legend(['Predicted new cases','Actual new cases'])
plt.show()

# print('Actual MAE error is {}'.format(mean_absolute_error(y_actual,y_pred)))
print('Actual MAPE error is {}'.format(mean_absolute_percentage_error(y_actual,y_pred)))
# print('Actual R2 value is {}'.format(r2_score(y_actual,y_pred)))

# %% transform to actual data
y_pred_iv = mms.inverse_transform(y_pred)
y_actual_iv = mms.inverse_transform(y_actual)

plt.figure()
plt.plot(y_pred_iv,color='red')
plt.plot(y_actual_iv,color='blue')
plt.legend(['Predicted new cases','Actual new cases'])
plt.show()

#%% model deployment
mms_path = os.path.join(os.getcwd(),'model','mms.pkl')
model_path = os.path.join(os.getcwd(),'model','model.h5')

model.save(model_path)

with open(mms_path,'wb') as f:
    pickle.dump(mms,f)

