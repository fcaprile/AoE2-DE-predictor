# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 15:41:58 2021

@author: Fernando Caprile
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import train_test_split
from preprocessing_features import prep
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

df=pd.read_csv('Processed_features.csv')
df.drop("Unnamed: 0",axis=1,inplace=True)
y=np.reshape(df['win1'].to_numpy(),(-1,1))
labels=[]
labels.append('elo_dif')

labels.append('tc_abs_distance')    
# labels.append('oro0_dist_TC_0')
# labels.append('oro0_dist_TC_1')
# labels.append('dif_dist_oro_mas_cercano')
# df['dif_dist_oro_mas_cercano']=df['oro0_dist_TC_0']-df['oro0_dist_TC_1']
# labels.append('menor_ang_oro_0')
# df['menor_ang_oro_0']=df[['oro0_angle_TC_0','oro1_angle_TC_0','oro2_angle_TC_0']].min(axis=1)
# labels.append('menor_ang_oro_1')
# df['menor_ang_oro_1']=df[['oro0_angle_TC_1','oro1_angle_TC_1','oro2_angle_TC_1']].min(axis=1)

# labels.append('dif_menor_ang_oro')
# df['dif_menor_ang_oro']=df[['oro0_angle_TC_0','oro1_angle_TC_0','oro2_angle_TC_0']].min(axis=1)-df[['oro0_angle_TC_1','oro1_angle_TC_1','oro2_angle_TC_1']].min(axis=1)

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


# for i in range(12):
#     labels.append('random'+str(i))
#     aux=np.random.uniform(size=len(df['elo_dif']))
#     df['random'+str(i)]=aux

model=KNC(10)
X=df[labels].to_numpy()
aucs=[]
plt.close('all')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
prep(2,X_train,X_test)
model.fit(X_train, np.ravel(y_train))
probs = model.predict_proba(X_test)
fpr, tpr, threshold = metrics.roc_curve(y_test, probs[:,1])        
auc = metrics.auc(fpr, tpr)
aucs.append(auc)
plt.plot(fpr, tpr, label = 'AUC = %0.2f' % auc)
plt.plot([0, 1], [0, 1],'r--',label='Random prediction')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid(True)
plt.legend(loc = 'lower right')
plt.tight_layout()

print(str(np.round(np.mean(aucs),3))+'+-'+str(np.round(np.std(aucs)/np.sqrt(3),3)))

