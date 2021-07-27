# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:34:40 2021

@author: ferchi
"""
# Imports de utilidades de Python
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Imports de Keras
from keras.datasets import mnist,fashion_mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils, plot_model
from keras.regularizers import l1
from keras.utils import to_categorical

# let's keep our keras backend tensorflow quiet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#column_names=('civ1','elo1','tc1','win1','civ2','elo2','tc2','win2','oros','piedras','bayas','bosques')
df=pd.read_csv('C:/Users/ferchi/Desktop/proyecto age/raw_features_sin_nan.csv')

y=df['win1']
'''
#For difference between elos, needs to add previous to split:
# X=np.reshape(X[:,0]-X[:,1],(-1,1))

#to add civilizations
# n_civs=np.max(df['civ1'].to_numpy())
# civs1=df['civ1'].to_numpy()
# civs1=np_utils.to_categorical(civs1-1, num_classes=n_civs,dtype='int64')
# civs2=df['civ2'].to_numpy()
# civs2=np_utils.to_categorical(civs2-1, num_classes=n_civs,dtype='int64')
# civ_labels=[]
# for i in range(2):
#     for j in range(n_civs):
#         civ_labels.append('civ'+str(i)+'_'+str(j))
# civ_labels=np.array(civ_labels)

# df_civs=pd.DataFrame(np.hstack([civs1,civs2]),columns=civ_labels)
# df_cat=pd.concat((df,df_civs))
# df_cat.drop(['civ1','civ2'],axis=1)
# X=df_cat.to_numpy()
'''

X=df['elo1','tc1','elo2','tc2','oros','piedras','bayas','bosques']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

min_max_scaler = preprocessing.MinMaxScaler()
X_train=min_max_scaler.fit_transform(X_train)
X_test=min_max_scaler.fit_transform(X_test)
y_train=to_categorical(np.reshape(np.array(y_train)*1,(-1,1)))
y_test=to_categorical(np.reshape(np.array(y_test)*1,(-1,1)))
# y_train=np.reshape(np.array(y_train)*1,(-1,1))
# y_test=np.reshape(np.array(y_test)*1,(-1,1))



model_simple = Sequential()
model_simple.add(Dense(2,activation='softmax'))
model_simple.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')

model_simple.fit(X_train, y_train,batch_size=32, epochs=10,verbose=1,validation_data=(X_test, y_test))

salida = model_simple.predict(X_test)<0.5*1

# print('Salida:',salida)
