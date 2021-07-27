# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:34:40 2021

@author: Fernando Caprile
"""
# Imports de utilidades de Python
import numpy as np
import pandas as pd

# let's keep our keras backend tensorflow quiet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

#column_names=('civ1','elo1','tc1_1','tc1_2','win1','civ2','elo2','tc2_1','tc2_2','win2',*labels_oros,*labels_piedras,*labels_bayas,*labels_bosques)        
df1=pd.read_csv('C:/Users/ferchi/Desktop/proyecto age/raw_features_1.csv')
df1.drop("Unnamed: 0",axis=1,inplace=True)
df2=pd.read_csv('C:/Users/ferchi/Desktop/proyecto age/raw_features_2.csv')
df2.drop("Unnamed: 0",axis=1,inplace=True)
# df3=pd.read_csv('C:/Users/ferchi/Desktop/proyecto age/raw_features_3.csv')
# df3.drop("Unnamed: 0",axis=1,inplace=True)

#df3 was seen to have the result switched (win1 = 1 to = 0 and viceversa)
# df3['win1']=(1-df3['win1'])
'''
#Check if an specific row is duplicated
N2=np.shape(df2)[0]
N_row=1
last=np.array(df1[['civ1','elo1','civ2','elo2']].iloc[-N_row])
for i in range(N2):
    if (np.array(df2[['civ1','elo1','civ2','elo2']].iloc[i])==last).all():
        print('Yep',i)
'''

df=pd.concat([df1,df2]).drop_duplicates(['civ1','elo1','civ2','elo2'],keep='last').reset_index(drop=True)
# df=pd.concat([df,df3]).drop_duplicates(['civ1','elo1','civ2','elo2'],keep='last').reset_index(drop=True)
df.dropna(inplace=True)

# df=df1.iloc[:3]

# X=np.array(df.drop(['win1','win2'],axis=1))
y=np.reshape(df['win1'].to_numpy(),(-1,1))
df=df[y==y]

df.to_csv('Total_raw_features.csv')
# df=df[X==X]
# y=y[np.where(y==y)[0]]

def normalize(x):
    return x/128

#Normalizo sobre el tamaño del mapa, 128
df[['tc1_1','tc1_2','tc2_1','tc2_2']]=df[['tc1_1','tc1_2','tc2_1','tc2_2']].apply(normalize)
# df[['tc1_1','tc1_2','tc2_1','tc2_2']]=df[['tc1_1','tc1_2','tc2_1','tc2_2']]/128


# X_raw=df[['elo1','tc1','elo2','tc2','oros','piedras','bayas','bosques']]
tc_distance_x=np.reshape((df['tc1_1']-df['tc2_1']).to_numpy(),(-1,1))#esta el tema de que als coordenadas de los tc van como x,y y el resto de las cosas como y,x
tc_distance_y=np.reshape((df['tc1_2']-df['tc2_2']).to_numpy(),(-1,1))
tc_abs_distance=(tc_distance_x**2+tc_distance_y**2)**0.5
# X_elos=df[['elo1','elo2']].to_numpy()
ELO_dif=np.reshape((df['elo1']-df['elo2']).to_numpy(),(-1,1))

labels_oros=[]
for i in range(8):
    for j in range(2):
        labels_oros.append('oro'+str(i)+'_'+str(j))
labels_piedras=[]
for i in range(6):
    for j in range(2):
        labels_piedras.append('piedra'+str(i)+'_'+str(j))
labels_bayas=[]
for i in range(2):
    for j in range(2):
        labels_bayas.append('baya'+str(i)+'_'+str(j))
labels_bosques=[]
for i in range(6):
    for j in range(2):
        labels_bosques.append('bosque'+str(i)+'_'+str(j))

df[labels_oros+labels_piedras+labels_bayas+labels_bosques]=df[labels_oros+labels_piedras+labels_bayas+labels_bosques].apply(normalize)
df.to_csv('C:/Users/ferchi/Desktop/proyecto age/normalized_features.csv')


N=np.shape(df)[0]

def get_closest(n,objetos):
    '''
    Get the n closest objects to the tc1 and tc2
    
    Args:
        objetos (tuple of dataframe columns): even elements from tuple are the x coordinate, odd elements the y coordintate. Each element of the array is made by each game recording.
    '''
    objetos=objetos.to_numpy()
    tc1_y=np.reshape(df['tc1_2'].to_numpy(),(-1,1))#esta el tema de que als coordenadas de los tc van como x,y y el resto de las cosas como y,x
    tc1_x=np.reshape(df['tc1_1'].to_numpy(),(-1,1))
    tc2_y=np.reshape(df['tc2_2'].to_numpy(),(-1,1))
    tc2_x=np.reshape(df['tc2_1'].to_numpy(),(-1,1))
    tc1_y_dist=objetos[:,::2]-tc1_y#se hace bien lo que quiero
    tc1_x_dist=objetos[:,1::2]-tc1_x
    tc2_y_dist=objetos[:,::2]-tc2_y
    tc2_x_dist=objetos[:,1::2]-tc2_x
    module_distance_tc1=np.array((tc1_y_dist**2+tc1_x_dist**2)**0.5)
    module_distance_tc2=np.array((tc2_y_dist**2+tc2_x_dist**2)**0.5)
    
    aux1=np.zeros((N,int(np.shape(objetos)[1]/2)))
    aux2=np.copy(aux1)
    aux3=np.copy(aux1)
    aux4=np.copy(aux1)
    for i in range(int(np.shape(objetos)[1]/2)):
        aux1[:,i]=tc1_y_dist[:,i]*np.transpose(tc_distance_y)
        aux2[:,i]=tc2_y_dist[:,i]*np.transpose(tc_distance_y)
        aux3[:,i]=tc1_x_dist[:,i]*np.transpose(tc_distance_x)
        aux4[:,i]=tc2_x_dist[:,i]*np.transpose(tc_distance_x)
    
    angle_from_tc1_to_tc2=np.arccos((aux1+aux3)/((tc1_y_dist**2+tc1_x_dist**2)**0.5*(tc_abs_distance)))
    angle_from_tc2_to_tc1=np.arccos((aux2+aux4)/((tc2_y_dist**2+tc2_x_dist**2)**0.5*(tc_abs_distance)))

    closest_tc1=np.zeros((N,n))
    closest_tc2=np.zeros((N,n))
    angles_closest_tc1=np.zeros((N,n))
    angles_closest_tc2=np.zeros((N,n))
    index_closest_tc1=np.zeros(n,dtype='int')
    index_closest_tc2=np.zeros(n,dtype='int')
    for i in range(N):
        closest_tc1[i,:]=sorted(module_distance_tc1[i,:])[:n]
        closest_tc2[i,:]=sorted(module_distance_tc2[i,:])[:n]
        for j in range(n):
            try:
                index_closest_tc1[j]=int(np.where(module_distance_tc1[i,:]==closest_tc1[i,j])[0])
            except: #to account for repeated distances
                index_closest_tc1[j]=int(np.where(module_distance_tc1[i,:]==closest_tc1[i,j])[0][0])
            try:
                index_closest_tc2[j]=int(np.where(module_distance_tc2[i,:]==closest_tc2[i,j])[0])
            except: #to account for repeated distances
                index_closest_tc2[j]=int(np.where(module_distance_tc2[i,:]==closest_tc2[i,j])[0][0])
        angles_closest_tc1[i,:]=angle_from_tc1_to_tc2[i,index_closest_tc1]/np.pi
        angles_closest_tc2[i,:]=angle_from_tc2_to_tc1[i,index_closest_tc2]/np.pi
        
    return closest_tc1,closest_tc2,angles_closest_tc1,angles_closest_tc2

closest_oros_tc1,closest_oros_tc2,angles_oros_tc1,angles_oros_2=get_closest(3,(df[labels_oros]))
closest_piedras_tc1,closest_piedras_tc2,angles_piedras_tc1,angles_piedras_2=get_closest(2,(df[labels_piedras]))
closest_bayas_tc1,closest_bayas_tc2,angles_bayas_tc1,angles_bayas_2=get_closest(1,(df[labels_bayas]))
closest_bosques_tc1,closest_bosques_tc2,angles_bosques_tc1,angles_bosques_2=get_closest(3,(df[labels_bosques]))

#make X features and y tags
X=np.hstack((ELO_dif,tc_abs_distance))

#add oros
X=np.hstack((X,closest_oros_tc1,closest_oros_tc2,angles_oros_tc1,angles_oros_2))
# X=np.hstack((X,angles_oros_tc1,angles_oros_2))

#piedras
X=np.hstack((X,closest_piedras_tc1,closest_piedras_tc2,angles_piedras_tc1,angles_piedras_2))

#bayas
X=np.hstack((X,closest_bayas_tc1,closest_bayas_tc2,angles_bayas_tc1,angles_bayas_2))

#bosques
X=np.hstack((X,closest_bosques_tc1,closest_bosques_tc2,angles_bosques_tc1,angles_bosques_2))
# X=np.hstack((X,angles_bosques_tc1,angles_bosques_2))

y=y[~np.isnan(X).any(axis=1)]
X=X[~np.isnan(X).any(axis=1)]

labels=[]
labels.append('win1')
labels.append('elo_dif')
labels.append('tc_abs_distance')
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


df_processed=pd.DataFrame(np.hstack((y,X)),columns=labels)

df_processed.to_csv('Processed_initial_features.csv')
# np.savetxt('C:/Users/ferchi/Desktop/proyecto age/module_angle_features.csv',X)


