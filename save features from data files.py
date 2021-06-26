# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 20:47:36 2021

@author: Fernando Caprile
"""

from matplotlib import pyplot as plt
from get_features_from_map import get_features 
import pandas as pd

civ_dict=dict({'Aztecs':14,'Berbers':24,'Britons':1,'Bulgarians':32,'Burmese':28,'Byzantines':2,'Celts':3,'Chinese':4,'Cumans':33,'Ethiopians':25,'Franks':5,'Goths':6,'Huns':15,'Incas':19,'Indians':20,'Italians':21,'Japanese':7,'Khmer':29,'Koreans':16,'Lithuanians':34,'Magyars':22,'Malay':30,'Malians':26,'Mayans':17,'Mongols':8,'Persians':9,'Portuguese':27,'Saracens':10,'Slavs':23,'Spanish':18,'Tatars':35,'Teutons':11,'Turks':12,'Vietnamise':31,'Vikings':13})

# AgeIIDE_Replay_100552757 tiene que dar malies y britanicos bling vs T90

folder='C:/Users/ferchi/Desktop/proyecto age/aoe recs/'
filename='AgeIIDE_Replay_100006892.aoe2record'
filename='AgeIIDE_Replay_100552757.aoe2record'
# filename='MP Replay v101.101.47820.0 @2021.06.07 220925 (2).aoe2record'
path=folder+filename

plt.close('all')

#get data from game recording
name1,civ1,tc1,win1,name2,civ2,tc2,win2,coord_oros,coord_piedras,coord_bayas,coord_bosques=get_features(path)

#get data from webpage
web_data=pd.read_csv('web_data.csv')

