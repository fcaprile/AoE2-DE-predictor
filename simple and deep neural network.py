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

# Imports de Keras
from keras.datasets import mnist,fashion_mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils, plot_model
from keras.regularizers import l1

# let's keep our keras backend tensorflow quiet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#column_names=('civ1','elo1','tc1','win1','civ2','elo2','tc2','win2','oros','piedras','bayas','bosques')
df=pd.read_csv('C:/Users/ferchi/Desktop/proyecto age/raw_features.csv')
df.dropna(inplace=True)

y=df['win1']
X=df['civ1','elo1','tc1','civ2','elo2','tc2','oros','piedras','bayas','bosques']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
np_utils.to_categorical(y_train, 35)
# X_train=




