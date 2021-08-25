# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:34:40 2021

@author: Fernando Caprile

Idea: pasar de 2D a 1D con una biyectiva y normalizar no max_min separado sino todos juntos por el mismo numero

Se observó que incluir mayores potencias de features resulta detrimental para el modelado, rduciendo la auc
"""
# Imports de utilidades de Python
import numpy as np
import pandas as pd
from deep_networks import simple, deep_1_layer,deep_2_layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from preprocessing_features import prep
import sklearn.metrics as metrics


import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


labels=[]
labels.append('elo_dif')
for tc in range(2):
    for nro in range(3):
        labels.append('oro'+str(nro)+'_dist_TC_'+str(tc))
for tc in range(2):
    for nro in range(3):
        labels.append('oro'+str(nro)+'_angle_TC_'+str(tc))

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
for tc in range(2):
    for nro in range(3):
        labels.append('bosque'+str(nro)+'_angle_TC_'+str(tc))

labels_todos=np.copy(labels)

labels=[]
labels.append('elo_dif')
for tc in range(2):
    for nro in range(3):
        labels.append('oro'+str(nro)+'_angle_TC_'+str(tc))
for tc in range(2):
    for nro in range(3):
        labels.append('bosque'+str(nro)+'_angle_TC_'+str(tc))

labels_ang_oros_bosques=np.copy(labels)

labels=[]
labels.append('elo_dif')

labels_elo_dif=np.copy(labels)

# df=pd.read_csv('Total_raw_features.csv')
# df.drop("Unnamed: 0",axis=1,inplace=True)
# X=np.array(df.drop(['win1','win2','civ1','civ2'],axis=1))

# X=np.array(df.drop(['win1','win2'],axis=1))
# X=np.reshape(df['elo_dif'].to_numpy(),(-1,1))
# X=np.reshape((df['elo1']-df['elo2']).to_numpy(),(-1,1))


# df=pd.read_csv('Processed_features.csv')

# df=pd.read_csv('Processed_initial_features.csv')
# df.drop("Unnamed: 0",axis=1,inplace=True)
# X=np.array(df.drop(['win1'],axis=1))

# y=np.reshape(df['win1'].to_numpy(),(-1,1))


prep_kind=2


# df_raw=pd.read_csv('Processed_features.csv')
df_raw=pd.read_csv('Processed_initial_features.csv')
df_raw.drop("Unnamed: 0",axis=1,inplace=True)

limit=20
df=df_raw[np.abs(df_raw['elo_dif'])<limit]

y=np.reshape(df['win1'].to_numpy(),(-1,1))

# X_temp=df[labels_elo_dif].to_numpy()
# X_temp=df[labels_ang_oros_bosques].to_numpy()
X_temp=df[labels_todos].to_numpy()
X=np.copy(X_temp)
N_features=np.shape(X)[1]

model_number=0
if model_number==0:
    model=simple(N_features)
if model_number==1:
    model=deep_1_layer(N_features)
if model_number==2:
    model=deep_2_layers(N_features)

kfold = KFold(n_splits=5, shuffle=True)

plt.close('all')
    
roc_aucs=[]
for train_index, test_index in kfold.split(X,y):
    X_train,y_train=X[train_index], y[train_index]
    X_test,y_test=X[test_index], y[test_index]
    prep(prep_kind,X_train,X_test)
    model.fit(X_train, y_train,batch_size=50,callbacks=[], epochs=15,verbose=0)
    probs = model.predict_proba(X_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    
    roc_auc = metrics.auc(fpr, tpr)
    roc_aucs.append(roc_auc)
    
    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)

plt.plot([0, 1], [0, 1],'r--',label='Random prediction')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid(True)
plt.legend(loc = 'lower right')
plt.tight_layout()


    