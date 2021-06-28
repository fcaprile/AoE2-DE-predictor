# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 20:47:36 2021

@author: Fernando Caprile
"""

from matplotlib import pyplot as plt
from get_features_from_map import get_features 
import pandas as pd
import os 
import numpy as np
from tqdm import tqdm

civ_dict=dict({'Aztecs':14,'Berbers':24,'Britons':1,'Bulgarians':32,'Burmese':28,'Byzantines':2,'Celts':3,'Chinese':4,'Cumans':33,'Ethiopians':25,'Franks':5,'Goths':6,'Huns':15,'Incas':19,'Indians':20,'Italians':21,'Japanese':7,'Khmer':29,'Koreans':16,'Lithuanians':34,'Magyars':22,'Malay':30,'Malians':26,'Mayans':17,'Mongols':8,'Persians':9,'Portuguese':27,'Saracens':10,'Slavs':23,'Spanish':18,'Tatars':35,'Teutons':11,'Turks':12,'Vietnamise':31,'Vikings':13})

carpeta_rec='C:/Users/ferchi/Desktop/proyecto age/save files/'
lista=[]
for archivo in os.listdir(carpeta_rec):
    lista.append(archivo)

plt.close('all')
features=[]

#get data from webpage
#name1,civ1,elo1,name2,civ2,elo2
df=pd.read_csv('C:/Users/ferchi/Desktop/proyecto age/elo index.csv')

N_data=np.shape(df)[0]
# N_data=100
N_previous=0
for i in tqdm(range(N_data-N_previous)):
    i+=N_previous
    #get data from game recording
    path=carpeta_rec+lista[i]
    try:
        name1,civ1,tc1,win1,name2,civ2,tc2,win2,coord_oros,coord_piedras,coord_bayas,coord_bosques=get_features(path)
    except:
        pass
    '''    
    #asociate elo with player, after many test it was observed that all files have the elos switched from the web page, meaning elo1=df['elo2'][i], elo2=df['elo1'][i]

    if civ1!=civ2:
        if civ1==civ_dict[df['civ1'][i]]:
            elo1=df['elo1'][i]
            elo2=df['elo2'][i]
            print('1')
        elif civ1==civ_dict[df['civ2'][i]]:
            elo1=df['elo2'][i]
            elo2=df['elo1'][i]
            print('2')
        else:
            print('Alvo va mal en iteración', i)
            continue
    else:
    if name1==df['name1'][i] or name2==df['name2'][i]:
        elo1=df['elo1'][i]
        elo2=df['elo2'][i]
        N1+=1
    else:
        if name2==df['name1'][i] or name1==df['name2'][i]:
            elo1=df['elo2'][i]
            elo2=df['elo1'][i]
            N2+=1
            # else:
            #     continue
print('Casos 1:',N1/(N1+N2))
print('Casos 2:',N2/(N1+N2))
    '''   
    if name1==df['name1'][i] and name2==df['name2'][i]:#just in case it does not happen in every match
        elo1=df['elo1'][i]
        elo2=df['elo2'][i]
    else:
        elo1=df['elo2'][i]
        elo2=df['elo1'][i]
    features.append((civ1,elo1,tc1,win1,civ2,elo2,tc2,win2,coord_oros,coord_piedras,coord_bayas,coord_bosques))
        
=df=pd.DataFrame(features,columns=column_names)
df.to_csv('C:/Users/ferchi/Desktop/proyecto age/raw_features.csv')
# print('Partidas leídas: ',len(data))


#to do: checkear que estan en orden, o sea comparar nombres y si le empieo a pifiar a todas probar con +1 o -1
