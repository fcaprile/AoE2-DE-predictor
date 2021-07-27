# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:34:40 2021

@author: Fernando Caprile


Idea: pasar de 2D a 1D con una biyectiva y normalizar no max_min separado sino todos juntos por el mismo numero
"""
# Imports de utilidades de Python
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
# Imports de Keras

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# let's keep our keras backend tensorflow quiet
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

labels=[]
labels.append('win1')
for tc in range(2):
    for nro in range(3):
        labels.append('oro'+str(nro)+'_dist_TC_'+str(tc))
# for tc in range(2):
#     for nro in range(3):
#         labels.append('oro'+str(nro)+'_angle_TC_'+str(tc))

for tc in range(2):
    for nro in range(2):
        labels.append('piedra'+str(nro)+'_dist_TC_'+str(tc))
for tc in range(2):
    for nro in range(2):
        labels.append('piedra'+str(nro)+'_angle_TC_'+str(tc))

for tc in range(2):
    for nro in range(1):
        labels.append('baya'+str(nro)+'_dist_TC_'+str(tc))
for tc in range(2):
    for nro in range(1):
        labels.append('baya'+str(nro)+'_angle_TC_'+str(tc))

for tc in range(2):
    for nro in range(3):
        labels.append('bosque'+str(nro)+'_dist_TC_'+str(tc))
# for tc in range(2):
#     for nro in range(3):
#         labels.append('bosque'+str(nro)+'_angle_TC_'+str(tc))




# df=pd.read_csv('Processed_features.csv')

df=pd.read_csv('Processed_initial_features.csv')
df.drop("Unnamed: 0",axis=1,inplace=True)
# X=np.array(df.drop(['win1'],axis=1))
X=pd.DataFrame.copy(df)
for label in labels:
    X.drop([label],axis=1,inplace=True)
X=X.to_numpy()

# df=pd.read_csv('Total_raw_features.csv')
# df.drop("Unnamed: 0",axis=1,inplace=True)
# X=np.array(df.drop(['win1','win2','civ1','civ2'],axis=1))

# X=np.array(df.drop(['win1','win2'],axis=1))
# X=np.reshape(df['elo_dif'].to_numpy(),(-1,1))
# X=np.reshape((df['elo1']-df['elo2']).to_numpy(),(-1,1))


y=np.reshape(df['win1'].to_numpy(),(-1,1))

# X=np.hstack((X,X**2,X**3,X**4,X**5))
# X=np.hstack((X,X**2,X**3))
# X=np.hstack((X,X**2))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

min_max_scaler = preprocessing.MinMaxScaler()
# X_train=min_max_scaler.fit_transform(X_train)
# X_test=min_max_scaler.fit_transform(X_test)

# y_train=np.reshape(y_train,(-1,1))
# y_test=np.reshape(y_test,(-1,1))
# y_train=to_categorical(y_train)
# y_test=to_categorical(y_test)

# y_train=np.reshape(np.array(y_train)*1,(-1,1))
# y_test=np.reshape(np.array(y_test)*1,(-1,1))

def plot_roc_curve(model,X_test,y_test,points=20):
    probs = model.predict_proba(X_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr)
    return roc_auc
    
# class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train[:,0])
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(np.shape(X)[1], input_dim=np.shape(X)[1], activation='relu'))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# history=model.fit(X_train, y_train,batch_size=32,callbacks=[], epochs=15,verbose=1,validation_data=(X_test, y_test))

estimator = KerasClassifier(build_fn=create_baseline, epochs=10, batch_size=50, verbose=0)
kfold = KFold(n_splits=5, shuffle=True)
# results = cross_val_score(estimator, X, y, cv=kfold)
plt.close('all')
start=time.time()
for train_index, test_index in kfold.split(X,y):
    model=create_baseline()
    X_train,y_train=X[train_index], y[train_index]
    X_test,y_test=X[test_index], y[test_index]
    # X_train=min_max_scaler.fit_transform(X_train)
    # X_test=min_max_scaler.fit_transform(X_test)
    model.fit(X_train, y_train,batch_size=50,callbacks=[], epochs=15,verbose=0)
    plot_roc_curve(model,X_test,y_test)
print('Tiempo transcurrido:',time.time()-start)

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid(True)
    
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# print(model.evaluate(X_test, y_test,verbose=0)[1])

# plot_roc_curve(model,X_test, y_test)

'''old custom plot roc_curve
def plot_roc_curve(model,X_test,y_test,points=20):
    true_positive=np.zeros(points)
    false_positive=np.zeros(points)
    cuts=np.linspace(1/points,1,points)
    for i,cut in enumerate(cuts):
        salida = model.predict_proba(X_test)<cut*1
        true_positive[i]=confusion_matrix(y_test,salida)[0,1]
        false_positive[i]=confusion_matrix(y_test,salida)[1,1]
    plt.plot(false_positive,true_positive)
'''
    
