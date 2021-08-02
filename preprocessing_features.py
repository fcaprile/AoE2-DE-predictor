# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 15:29:43 2021

@author: Fernando Caprile
"""
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()        
standard_scaler = preprocessing.MinMaxScaler()        

def prep(prep_kind,X_train,X_test):
    '''
    prep_kind = 0 for no processing
    prep_kind = 1 for min_max
    prep_kind = 2 for starndard (Z-score)
    '''
    if prep_kind==0:
        pass
    if prep_kind==1:    
        X_train=min_max_scaler.fit_transform(X_train)
        X_test=min_max_scaler.fit_transform(X_test)
    if prep_kind==1:    
        X_train=standard_scaler.fit_transform(X_train)
        X_test=standard_scaler.fit_transform(X_test)
        
    return X_train,X_test
