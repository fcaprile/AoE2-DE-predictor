# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:06:46 2021

@author: ferchi
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


df=pd.read_csv('Processed_features.csv')
elodif=df['elo_dif']

# df1=pd.read_csv('C:/Users/ferchi/Desktop/proyecto age/raw_features_1.csv')
# df2=pd.read_csv('C:/Users/ferchi/Desktop/proyecto age/raw_features_2.csv')
# df3=pd.read_csv('C:/Users/ferchi/Desktop/proyecto age/raw_features_3.csv')
# df3['win1']=(1-df3['win1'])

# df=pd.concat([df1,df2]).drop_duplicates(['civ1','elo1','civ2','elo2'],keep='last').reset_index(drop=True)
# df=pd.concat([df,df3]).drop_duplicates(['civ1','elo1','civ2','elo2'],keep='last').reset_index(drop=True)
# df.dropna(inplace=True)

# elodif=(df['elo1']-df['elo2']).to_numpy()

# ELO_max=2300
# ELO_min=1800
# elodif=elodif[(df['elo1']<ELO_max) & (df['elo1']>ELO_min) & (df['elo2']<ELO_max) & (df['elo2']>ELO_min)]
# df=df[(df['elo1']<ELO_max) & (df['elo1']>ELO_min) & (df['elo2']<ELO_max) & (df['elo2']>ELO_min)]

N_bins=15
elodif_counts,elodif_labels=pd.qcut(elodif,N_bins,labels=range(N_bins),retbins=True)

winrates=np.zeros(N_bins)
for i in range(N_bins):
    winrates[i]=sum(elodif_counts[df['win1']==True]==i)/sum(elodif_counts==i)#count the number of times the elo difference was in the quantile number i and player 1 won, then divide the number of times the elo difference was in the quantile number i 

winrate_labels=np.zeros(N_bins)
for i in range(N_bins):
    winrate_labels[i]=(elodif_labels[i]+elodif_labels[i+1])/2
    
    
plt.close('all')
f= lambda x,b: 0.5+b*x
parametros_optimizados, matriz_covarianza = curve_fit(f,winrate_labels[1:-1],winrates[1:-1]) 

x_axis=np.linspace(np.min(winrate_labels),np.max(winrate_labels),1000)
    
plt.plot(winrate_labels, winrates,'b*')
plt.plot(x_axis,f(x_axis,*parametros_optimizados))
plt.ylabel('Winrate')
plt.xlabel('ELO difference')
plt.grid(True)

print(parametros_optimizados)
'''
The winrate against ELO difference has the correct limits and is close to the expected values of 35% winrate at a -100 elo differnce and 65% winreate at 100 elo difference
'''
