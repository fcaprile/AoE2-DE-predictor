# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 13:39:25 2021

@author: Fernando Caprile
"""
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def plot_roc_curve(model,X_test,y_test,points=20,save_name=''):
        
    return roc_auc

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
    
