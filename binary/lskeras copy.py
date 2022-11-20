from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

df = pd.read_csv('train.csv', header=None)
# testdata = pd.read_csv('test.csv', header=None)


# def encode_numeric_zscore(df, name, mean=None, sd=None):
#     if mean is None:
#         mean = df[name].mean()

#     if sd is None:
#         sd = df[name].std()

#     df[name] = (df[name] - mean) / sd
    
# # Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] 
# # for red,green,blue)
# def encode_text_dummy(df, name):
#     dummies = pd.get_dummies(df[name])
#     for x in dummies.columns:
#         dummy_name = f"{name}-{x}"
#         df[dummy_name] = dummies[x]
#     df.drop(name, axis=1, inplace=True)

# for name in df.columns:
#     if name == 'outcome':
#         pass
#     elif name in ['protocol_type','service','flag','land','logged_in',
#                     'is_host_login','is_guest_login']:
#         encode_text_dummy(df,name)
#     else: 
#         encode_numeric_zscore(df,name)    

normal_mask = df['label']=='normal.'
attack_mask = df['label']!='normal.'

df.drop('label',axis=1,inplace=True)

df_normal = df[normal_mask]
df_attack = df[attack_mask]
# This is the numeric feature vector, as it goes to the neural net
x_normal = df_normal.values
x_attack = df_attack.values

from sklearn.model_selection import train_test_split

x_normal_train, x_normal_test = train_test_split(
    x_normal, test_size=0.25, random_state=42)

x_normal_train.to_csv('raw_data.csv', index=False)

X = x_normal_train.iloc[:,1:42]
Y = x_normal_train.iloc[:,0]
C = x_normal_test.iloc[:,0]
T = x_normal_test.iloc[:,1:42]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
#print(trainX[0:5,:])

scaler = Normalizer().fit(T)
testT = scaler.transform(T)
# summarize transformed data
np.set_printoptions(precision=3)
#print(testT[0:5,:])


y_train = np.array(Y)
y_test = np.array(C)


# reshape input to be [samples, time steps, features]
X_train = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
X_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))


print(X_train.shape)


batch_size = 32

# 1. define the network
model = Sequential()
model.add(LSTM(4,input_dim=41))  # try using a GRU instead, for fun
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('sigmoid'))
print(model.get_config())


# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/lstm1layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
csv_logger = CSVLogger('training_set_iranalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size,epochs=20, validation_data=(X_test, y_test),callbacks=[checkpointer,csv_logger])
model.save("kddresults/lstm1layer/fullmodel/lstm1layer_model.hdf5")

loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
# y_pred = model.predict(X_test)
y_pred = np.argmax(model.predict(X_test), axis=-1)
np.savetxt('kddresults/lstm1layer/lstm1predicted.txt', y_pred, fmt='%01d')







