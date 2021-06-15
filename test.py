# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 18:55:25 2021

@author: ferchi con ayuda de matiu

factores a agregar:
    factor de peligro de oro
    factor de peligro de piedra
    distancia entre tcs
    distancia entre tc y cada madera

"""


from matplotlib import pyplot as plt
from mgz.summary import Summary
import numpy as np

folder='C:/Users/ferchi/Desktop/aoe scripts/recs/'
filename='Scout seleccionado y movido.aoe2record'
filename='MP Replay v101.101.47820.0 @2021.06.06 234535 (2)'+'.aoe2record'
path=folder+filename

with open(path, 'rb') as data:
    s = Summary(data)
    mapa=s.get_map()
    tiles=mapa['tiles']
    plataforma=s.get_platform()
    objetos=s.get_objects()['objects']
#%%
plt.close('all')
obj=np.zeros((120,120))

'''
oro: 66
piedra: 102
TC:1649
bayas: 1059


aldeanos: 83 (Hombre) 293 (Mujer)
reliquia: 285
obejas: 1060
boar: 48
scout:751 (aguila) 448 (scout)
'''

for diccionario in objetos:
    x=int(diccionario['x'])
    y=int(diccionario['y'])
    if not diccionario['object_id']==647:
        obj[y,x]=diccionario['object_id']
    # if diccionario['object_id']>=1350:
        # obj[y,x]=diccionario['object_id']
    

fig = plt.figure(figsize=(8, 8))
axis=[0,120,0,120]
plt.imshow(obj,extent=axis, interpolation='none', aspect='equal')



# fig3 = plt.figure(figsize=(8, 4))
# N_ids=2000
# occurrences=np.zeros(N_ids)
# for i in range(N_ids):
#     occurrences[i] = np.count_nonzero(obj == i)

# plt.bar(np.arange(0,N_ids),occurrences)

'''    
plt.close('all')
terreno=np.zeros((120,120))
elevacion=np.zeros((120,120))
# ids=[]
for diccionario in tiles:
    x=int(diccionario['x'])
    y=int(diccionario['y'])
    terreno[x,y]=diccionario['terrain_id']
    elevacion[x,y]=diccionario['elevation']
    # ids.append(diccionario['terrain_id'])
    
fig = plt.figure(figsize=(8, 8))
axis=[0,120,0,120]
plt.imshow(np.transpose(terreno),extent=axis, interpolation='none', aspect='equal')
   
# fig2 = plt.figure(figsize=(8, 8))
# plt.imshow(np.transpose(elevacion),extent=axis, interpolation='none', aspect='equal')

fig3 = plt.figure(figsize=(8, 4))
N_ids=100
occurrences=np.zeros(N_ids)
for i in range(N_ids):
    occurrences[i] = np.count_nonzero(terreno == i)

plt.bar(np.arange(0,N_ids),occurrences)
'''