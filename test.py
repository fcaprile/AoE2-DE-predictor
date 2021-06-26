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
from sklearn.cluster import KMeans
from find_cluster_center import find_cluster_center
from sklearn.cluster import AffinityPropagation

folder='C:/Users/ferchi/Desktop/proyecto age/aoe recs/'
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

''' Object ids importantes:
oro: 66
piedra: 102
TC:1649
bayas: 1059


aldeanos: 83 (Hombre) 293 (Mujer)
reliquia: 285
ovejas: 1060
boar: 48
scout:751 (aguila) 448 (scout)

Terrain ids:
bosque:13
'''

piedras=[]
oros=[]
TCs=[]
bayas=[]

for diccionario in objetos:
    x=int(diccionario['x'])
    y=int(diccionario['y'])
    if diccionario['object_id']!=647:#saco los circulos donde pueden spawnear las ovejas, es algo interno del juego
        obj[y,x]=diccionario['object_id']
    if diccionario['object_id']==66:
        oros.append([y,x])
    if diccionario['object_id']==102:
        piedras.append([y,x])
    if diccionario['object_id']==1649:
        TCs.append([y,x])
    if diccionario['object_id']==1059:
        bayas.append([y,x])

oros=np.array(oros)
piedras=np.array(piedras)
coord_TCs=np.array(TCs)
bayas=np.array(bayas)

coord_oros=find_cluster_center(oros, 8)
coord_piedras=find_cluster_center(piedras, 6)
coord_bayas=find_cluster_center(bayas, 2)


bosques=[]
terreno=np.zeros((120,120))
d_bosque_TC=30#maxima distancia entre bosque y tc para que sea uno de los bosques iniciales
# elevacion=np.zeros((120,120))
for diccionario in tiles:
    x=int(diccionario['x'])
    y=int(diccionario['y'])
    if diccionario['terrain_id']==13:
        if ((y-coord_TCs[0,0])**2+(x-coord_TCs[0,1])**2)**0.5<d_bosque_TC or ((y-coord_TCs[1,0])**2+(x-coord_TCs[1,1])**2)**0.5<d_bosque_TC:
            bosques.append([y,x])
    terreno[y,x]=diccionario['terrain_id']
    # elevacion[x,y]=diccionario['elevation']

coord_bosques=find_cluster_center(bosques, 6)

axis=[0,120,0,120]
plt.figure(figsize=(8,8))
plt.imshow(terreno,extent=axis, interpolation='none', aspect='equal')

# preferences=find_cluster_center(bosques, 22)
# cluster_finder=AffinityPropagation(random_state=0,max_iter=100,preference=preferences)
# cluster_finder.fit(bosques)
# coord_bosques=cluster_finder.cluster_centers_

plt.scatter(coord_bosques[:, 1], 120-coord_bosques[:, 0], c='green', s=100, alpha=0.5)



