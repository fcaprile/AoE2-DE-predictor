# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 13:39:00 2021

@author: Fernando Caprile
"""

from keras.models import Sequential
from keras.layers import Dense

def simple(N_features):
    # create model
    model = Sequential()
    model.add(Dense(N_features, input_dim=N_features, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def deep_1_layer(N_features):
    # create model
    model = Sequential()
    model.add(Dense(N_features, input_dim=N_features, activation='relu'))
    model.add(Dense(2*N_features, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def deep_2_layers(N_features):
    # create model
    model = Sequential()
    model.add(Dense(N_features, input_dim=N_features, activation='relu'))
    model.add(Dense(2*N_features, activation='relu'))
    model.add(Dense(2*N_features, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
