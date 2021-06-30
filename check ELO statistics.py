# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:06:46 2021

@author: ferchi
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


#column_names=('civ1','elo1','tc1','win1','civ2','elo2','tc2','win2','oros','piedras','bayas','bosques')
df=pd.read_csv('C:/Users/ferchi/Desktop/proyecto age/raw_features.csv')
df.dropna(inplace=True)
column_names=('civ1','elo1','tc1','win1','civ2','elo2','tc2','win2','oros','piedras','bayas','bosques')        
df=pd.DataFrame(df,columns=column_names)
df.to_csv('C:/Users/ferchi/Desktop/proyecto age/raw_features.csv')

elodif=df['elo1']-df['elo2']
ELO_max=2300
elodif=elodif[np.logical_and(df['elo1']<ELO_max, df['elo2']<ELO_max)]

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
parametros_optimizados, matriz_covarianza = curve_fit(f,winrate_labels[1:-2],winrates[1:-2]) 

x_axis=np.linspace(np.min(winrate_labels),np.max(winrate_labels),1000)
    
plt.plot(winrate_labels, winrates,'b*')
plt.plot(x_axis,f(x_axis,*parametros_optimizados))
plt.ylabel('Winrate')
plt.xlabel('ELO difference')
plt.grid(True)

'''
The winrate against ELO difference has the correct limits and is close to the expected values of 35% winrate at a -100 elo differnce and 65% winreate at 100 elo difference
'''
