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
# import numpy as np
import matplotlib.pyplot as plt
# import sklearn.metrics as metrics
# from sklearn.model_selection import train_test_split
# from sklearn.utils import class_weight
# from sklearn.metrics import confusion_matrix
# Imports de Keras

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# let's keep our keras backend tensorflow quiet

from train_and_evaluate_predictor import train_predictor_and_evaluate

import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

base_saving_folder='C:/Users/ferchi/Desktop/proyecto age/ROC curves/'

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


N_iteraciones=10
prep_kind=2

folder1='all datasets/'
folder2='initial 2 datasets/'
folder3='raw data/'
folders=(folder1,folder2)
datafiles=('Processed_features.csv','Processed_initial_features.csv')

model1='simple/'
model2='1 layer/'
model3='2 layers/'
models=(model1,model2)

name1='all features'
name2='gold and forest angles'
name3='only elo difference'
names=(name1,name2,name3)
labels=(labels_todos,labels_ang_oros_bosques,labels_elo_dif)

power1=' lineal'
power2=' cuadratico'
power3=' cubico'
powers=(power1,power2,power3)
p=0
roc_aucs=np.zeros((2*2*3,N_iteraciones*5))
path=''

for f,folder in enumerate(folders):
    df=pd.read_csv(datafiles[f])
    df.drop("Unnamed: 0",axis=1,inplace=True)
    y=np.reshape(df['win1'].to_numpy(),(-1,1))
    for m,model in enumerate(models):
        for n,name in enumerate(names):
            print(folder+model+name)
            label=labels[n]
            for i in range(N_iteraciones):
                # if i==0:
                #     path=base_saving_folder+folder+model+name
                # else:
                #     path=''
                a=f*2*3+m*3+n
                roc_aucs[a,i*5:(i+1)*5]=train_predictor_and_evaluate(m,df,label,y,prep_kind,p,path)

np.savetxt('roc_aucs.txt',roc_aucs)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# print(model.evaluate(X_test, y_test,verbose=0)[1])

# plot_roc_curve(model,X_test, y_test)

for a in range(2*2*3):
    aucs=roc_aucs[a,:]
    print(str(np.round(np.mean(aucs),3))+'+-'+str(np.round(np.std(aucs)/np.sqrt(N_iteraciones*5),3)))
    