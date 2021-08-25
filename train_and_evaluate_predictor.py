# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 15:41:58 2021

@author: Fernando Caprile
"""
from deep_networks import simple, deep_1_layer,deep_2_layers
from preprocessing_features import prep
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def train_predictor_and_evaluate(model_number,df,labels,y,prep_kind=2,power=0,save_path='',print_acc=False):
    # X=pd.DataFrame.copy(df)
    
    X_temp=df[labels].to_numpy()
    X=np.copy(X_temp)
    for i in np.arange(0,power):
        X=np.hstack((X,X_temp**(i+2)))
    N_features=np.shape(X)[1]
    
    if model_number==0:
        model=simple(N_features)
    if model_number==1:
        model=deep_1_layer(N_features)
    if model_number==2:
        model=deep_2_layers(N_features)
    
    kfold = KFold(n_splits=5, shuffle=True)
    
    if not save_path=='':    
        plt.close('all')
        
    roc_aucs=[]
    for train_index, test_index in kfold.split(X,y):
        X_train,y_train=X[train_index], y[train_index]
        X_test,y_test=X[test_index], y[test_index]
        prep(prep_kind,X_train,X_test)
        model.fit(X_train, y_train,batch_size=50,callbacks=[], epochs=15,verbose=0)
        probs = model.predict_proba(X_test)
        if print_acc:
            pos=probs>0.5
            acc=sum(pos==y_test)/len(y_test)
            print('Acuracy:', acc)
        fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
        
        roc_auc = metrics.auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        
        if not save_path=='':
            plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
    
    if not save_path=='':
        plt.plot([0, 1], [0, 1],'r--',label='Random prediction')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.grid(True)
        plt.legend(loc = 'lower right')
        plt.tight_layout()
        plt.savefig(save_path+'.jpg')

    return roc_aucs


if __name__=='__main__':
    import pandas as pd
    df=pd.read_csv('Processed_initial_features.csv')
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
    
    
    aucs=train_predictor_and_evaluate(1,df,labels,y,2,0,print_acc=False)
    print(str(np.round(np.mean(aucs),3))+'+-'+str(np.round(np.std(aucs)/np.sqrt(5),3)))

